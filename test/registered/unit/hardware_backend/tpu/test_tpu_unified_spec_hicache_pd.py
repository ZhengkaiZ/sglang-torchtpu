"""CPU unit tests for TPU unified KV pool speculative draft HiCache and equal-TP Raiden P/D."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.disaggregation.base.conn import KVPoll, KVTransferMetric, StateType
from sglang.srt.managers.cache_controller import BatchDoneCounter
from sglang.srt.mem_cache import hicache_auto_size
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import CacheOperation
from sglang.srt.mem_cache.pool_host.group import HostPoolGroup
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    HiCacheDraftMode,
    _can_pack_hicache_mtp,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang_tpu.disaggregation.raiden.common import (
    DRAFT_PORT_OFFSET,
    PORTS_PER_RANK,
    get_raiden_kv_caches,
)
from sglang_tpu.disaggregation.raiden.receiver import RaidenKVReceiver
from sglang_tpu.disaggregation.raiden.sender import RaidenKVSender
from sglang_tpu.disaggregation.raiden.serde import UNIFIED_SWA_KV_LAYOUT, ReshardReq
from sglang_tpu.hooks.draft_kv_pool import (
    bind_unified_swa_draft_host_pool,
    build_unified_hicache_draft_plan,
)
from sglang_tpu.kv_pool import (
    TpuBlockMajorMHATokenToKVPool,
    TpuFusedMHATokenToKVPool,
    TpuSeqAlongLaneKVPool,
)
from sglang_tpu.kv_pool_host import (
    TpuDMAUnifiedHostKVPool,
    _unified_swa_layer_mappings,
    register_unified_hicache_strategy,
    tpu_dma_host_kv_pool_cls,
)
from sglang_tpu.tpu_dma_hicache import page_major_tensors
from sglang_tpu.tpu_dma_hybrid_cache_controller import TpuDMAHybridCacheController
from sglang_tpu.unified_kv_pool import (
    AttnLayerSpec,
    TpuUnifiedSWAKVPool,
    TpuUnifiedSWATokenToKVPoolAllocator,
)

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

_spec = importlib.util.spec_from_file_location(
    "_disagg_raiden_transfer_harness_spec",
    pathlib.Path(__file__).resolve().parents[3]
    / "tpu/unit/test_disagg_raiden_transfer.py",
)
_harness = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _harness
_spec.loader.exec_module(_harness)
RaidenRendezvousTestCase = _harness.RaidenRendezvousTestCase
_fake_tpu_sync = _harness._fake_tpu_sync


class _DoneEvent:
    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        return None


class _InMemoryDMATransfer:
    def __init__(self, io_backend: str, device_pool, page_num: int, *, draft_pools=()):
        self.dev = [
            t for p in (device_pool, *draft_pools) for t in page_major_tensors(p)
        ]
        self.host = [
            torch.zeros((page_num, *t.shape[1:]), dtype=t.dtype) for t in self.dev
        ]

    def backup(self, device_pages, host_pages):
        for dev, host in zip(self.dev, self.host, strict=True):
            host[list(host_pages)] = dev[list(device_pages)]
        return _DoneEvent()

    def load(self, host_pages, device_pages):
        for dev, host in zip(self.dev, self.host, strict=True):
            dev[list(device_pages)] = host[list(host_pages)]
        return _DoneEvent()

    def destroy(self) -> None:
        return None


def _make_target(
    *,
    size: int = 64,
    full_layers: tuple[int, ...] = (0, 3),
    swa_layers: tuple[int, ...] = (1, 2, 4, 5, 6),
    pool_cls: type = TpuFusedMHATokenToKVPool,
) -> tuple[TpuUnifiedSWAKVPool, TpuUnifiedSWATokenToKVPoolAllocator]:
    pool = TpuUnifiedSWAKVPool(
        size=size,
        page_size=4,
        dtype=torch.bfloat16,
        full=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=full_layers),
        swa=AttnLayerSpec(head_num=2, head_dim=128, layer_ids=swa_layers),
        device="cpu",
        token_to_kv_pool_class=pool_cls,
    )
    alloc = TpuUnifiedSWATokenToKVPoolAllocator(
        size=size,
        page_size=4,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=pool,
        need_sort=False,
        sliding_window_size=8,
    )
    return pool, alloc


def _attach_draft(
    target_alloc: TpuUnifiedSWATokenToKVPoolAllocator,
    *,
    full_layers: tuple[int, ...],
    swa_layers: tuple[int, ...],
    head_dim: int = 64,
    pool_cls: type = TpuFusedMHATokenToKVPool,
):
    is_swa = bool(swa_layers)
    kvc = SimpleNamespace(
        is_hybrid_swa=is_swa,
        sliding_window_size=8 if is_swa else None,
        draft_swa_full_capacity=False,
        model_config=SimpleNamespace(sliding_window_size=None),
        layer_info=SimpleNamespace(
            full_attention_layer_ids=list(full_layers),
            swa_attention_layer_ids=list(swa_layers),
        ),
    )
    ft, df, ds = target_alloc.derive_draft_pool_sizes(
        kvc,
        max_total_num_tokens=target_alloc.size_full,
        full_max_total_num_tokens=target_alloc.size_full,
        swa_max_total_num_tokens=target_alloc.size_swa,
    )
    common = dict(
        page_size=4,
        dtype=torch.bfloat16,
        head_num=4,
        head_dim=head_dim,
        device="cpu",
        enable_memory_saver=False,
    )
    if not is_swa:
        pool = pool_cls(size=ft, layer_num=len(full_layers), **common)
    else:
        pool = SWAKVPool(
            size=df,
            size_swa=ds,
            swa_attention_layer_ids=list(swa_layers),
            full_attention_layer_ids=list(full_layers),
            token_to_kv_pool_class=pool_cls,
            **common,
        )
    target_alloc.attach_draft_kv_pool(kvc, pool)
    return pool


def _fill_pages(buffers, pages: list[int]) -> list[torch.Tensor]:
    out = []
    for buf in buffers:
        vals = torch.randn((len(pages), *buf.shape[1:]), dtype=buf.dtype)
        buf[pages] = vals
        out.append(vals.clone())
    return out


def _build_pair(pool: TpuUnifiedSWAKVPool, host_size: float = 0.0):
    return TpuDMAUnifiedHostKVPool.build_pair(
        pool,
        io_backend="raiden",
        host_to_device_ratio=2.0,
        host_size=host_size,
        page_size=4,
    )


def _translate_dswa(
    alloc: TpuUnifiedSWATokenToKVPoolAllocator, pages: list[int]
) -> list[int]:
    if alloc.draft_swa_sidecar is None:
        return pages
    return alloc.draft_swa_sidecar.translate_pages(torch.tensor(pages)).tolist()


class TestUnifiedSpecHiCacheSidecars(unittest.TestCase):
    def setUp(self) -> None:
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.addCleanup(reset_context)
        register_unified_hicache_strategy()
        orig_plan = BaseSpecWorker._build_hicache_draft_plan
        orig_swa = hybrid_pool_assembler.build_swa_draft_pools
        fake_mem = SimpleNamespace(
            hicache_io_backend="raiden",
            hicache_mem_layout="page_first",
            hicache_ratio=2.0,
            hicache_size=0.0,
        )
        for p in (
            patch.object(
                BaseSpecWorker,
                "_build_hicache_draft_plan",
                lambda w: build_unified_hicache_draft_plan(orig_plan, w),
            ),
            patch.object(
                hybrid_pool_assembler,
                "build_swa_draft_pools",
                lambda *a, **kw: bind_unified_swa_draft_host_pool(
                    orig_swa(*a, **kw), *a, **kw
                ),
            ),
            patch(
                "sglang_tpu.kv_pool_host.create_tpu_dma_kv_transfer",
                side_effect=_InMemoryDMATransfer,
            ),
            patch(
                "sglang_tpu.kv_pool_host.host_memory_budget_bytes", return_value=1 << 40
            ),
            patch(
                "sglang.srt.platforms.current_platform.get_mha_kv_pool_host_cls",
                side_effect=lambda *_, **__: tpu_dma_host_kv_pool_cls("raiden"),
            ),
            patch.object(hybrid_pool_assembler, "get_memory", return_value=fake_mem),
        ):
            p.start()
            self.addCleanup(p.stop)

    def _run_roundtrip(self, d_full: tuple[int, ...], d_swa: tuple[int, ...]) -> None:
        t_pool, t_alloc = _make_target(size=128)
        d_pool = _attach_draft(t_alloc, full_layers=d_full, swa_layers=d_swa)
        kv_host, swa_host = _build_pair(t_pool)
        full_map, swa_map = _unified_swa_layer_mappings(t_pool)
        max_lid = max(full_map.keys() | swa_map.keys()) + 1
        host_group = HostPoolGroup(
            [
                hybrid_pool_assembler.build_pool_entry(
                    name=PoolName.KV,
                    host_pool=kv_host,
                    device_pool=t_pool,
                    layer_mapping=full_map,
                    transfer_layer_id_max=max_lid,
                    is_anchor=True,
                ),
                hybrid_pool_assembler.build_pool_entry(
                    name=PoolName.SWA,
                    host_pool=swa_host,
                    device_pool=t_pool,
                    layer_mapping=swa_map,
                    transfer_layer_id_max=max_lid,
                    device_alloc_fn=t_alloc.swa_attn_allocator.alloc,
                    device_free_fn=t_alloc.swa_attn_allocator.free,
                ),
            ]
        )
        ctrl = object.__new__(TpuDMAHybridCacheController)
        ctrl.mem_pool_device_allocator, ctrl.mem_pool_host = t_alloc, host_group
        ctrl.page_size, ctrl.io_backend, ctrl.device = 4, "raiden", "cpu"
        ctrl.write_queue, ctrl.load_queue, ctrl.ack_write_queue, ctrl.ack_load_queue = (
            [],
            [],
            [],
            [],
        )
        ctrl.layer_done_counter = BatchDoneCounter()
        tree = SimpleNamespace(
            cache_controller=ctrl, register_sidecar_pools=lambda *_: None
        )
        specs, entries = hybrid_pool_assembler.build_hicache_draft_sidecars(
            draft_device_pools=(d_pool,), tree_cache=tree
        )
        for e in entries:
            host_group.add_entry(e)

        dev_full = t_alloc.full_attn_allocator.alloc(8)
        dev_swa = t_alloc.swa_attn_allocator.alloc(8)
        t_alloc.set_full_to_swa_mapping(dev_full, dev_swa)
        f_pages, s_prim = (dev_full[::4] // 4).tolist(), (dev_swa[::4] // 4).tolist()
        exp_tf = _fill_pages(t_pool.kv_pool.page_buffers(), f_pages)
        exp_ts = _fill_pages(
            t_pool.kv_pool.page_buffers(), t_pool.expand_swa_pages(s_prim)
        )
        d_sub = d_pool.swa_kv_pool if d_swa else d_pool
        d_pages = _translate_dswa(t_alloc, s_prim) if d_swa else f_pages
        exp_d = _fill_pages(page_major_tensors(d_sub), d_pages)

        host_full = host_group.alloc(8, pool=PoolName.KV)
        xfers = [PoolTransfer(name=PoolName.SWA, device_indices=dev_swa)] + [
            PoolTransfer(name=s.pool_name, indices_from_pool=s.indices_from_pool)
            for s in specs
        ]
        resolved = host_group.resolve_host_transfers(
            xfers, primary_device_indices=dev_full, primary_host_indices=host_full
        )
        host_swa = next(t.host_indices for t in resolved if t.name == PoolName.SWA)
        ctrl.write_queue.append(
            CacheOperation(host_full, dev_full, node_id=1, pool_transfers=resolved)
        )
        ctrl.start_writing()

        for buf in (*t_pool.kv_pool.page_buffers(), *page_major_tensors(d_sub)):
            buf.zero_()

        new_full = t_alloc.full_attn_allocator.alloc(8)
        new_swa = t_alloc.swa_attn_allocator.alloc(8)
        t_alloc.set_full_to_swa_mapping(new_full, new_swa)
        reload_xfers = [
            PoolTransfer(
                name=PoolName.SWA, host_indices=host_swa, device_indices=new_swa
            )
        ]
        for s in specs:
            h_idx, d_idx = (
                (host_full, new_full)
                if s.indices_from_pool == PoolName.KV
                else (host_swa, new_swa)
            )
            reload_xfers.append(
                PoolTransfer(
                    name=s.pool_name,
                    indices_from_pool=s.indices_from_pool,
                    host_indices=h_idx,
                    device_indices=d_idx,
                )
            )
        ctrl.load_queue.append(
            CacheOperation(host_full, new_full, node_id=2, pool_transfers=reload_xfers)
        )
        ctrl.start_loading()

        nf, ns = (new_full[::4] // 4).tolist(), (new_swa[::4] // 4).tolist()
        self.assertNotEqual(nf, f_pages)
        for buf, exp in zip(t_pool.kv_pool.page_buffers(), exp_tf):
            self.assertTrue(torch.equal(buf[nf], exp))
        for buf, exp in zip(t_pool.kv_pool.page_buffers(), exp_ts):
            self.assertTrue(torch.equal(buf[t_pool.expand_swa_pages(ns)], exp))
        nd = _translate_dswa(t_alloc, ns) if d_swa else nf
        for buf, exp in zip(page_major_tensors(d_sub), exp_d):
            self.assertTrue(torch.equal(buf[nd], exp))

        state = swa_host._state
        host_group.release_transfers(resolved)
        host_group.free(host_full, pool=PoolName.KV)
        self.assertEqual(
            len(state._compact_swa_free), state.page_num // state.num_swa_groups
        )
        self.assertEqual(state._host_to_compact_swa, {})

    def test_backup_evict_reload_and_host_free(self):
        self._run_roundtrip((0, 1), ())
        self._run_roundtrip((), (0, 1))

    def test_copricing_plan_and_edge_cases(self):
        with patch(
            "sglang_tpu.kv_pool_host.sync_fixed_hicache_size",
            side_effect=lambda t, _: t,
        ):
            p_plain, _ = _make_target()
            kv_plain, _ = _build_pair(p_plain, host_size=0.01)
            for full_l, swa_l in (((), (0, 1, 2, 3)), ((0, 1), ())):
                p_sp, a_sp = _make_target()
                d_sp = _attach_draft(
                    a_sp, full_layers=full_l, swa_layers=swa_l, head_dim=128
                )
                kv_sp, _ = _build_pair(p_sp, host_size=0.01)
                self.assertLess(kv_sp.page_num, kv_plain.page_num)
                exp = (
                    hicache_auto_size._pool_bytes(d_sp.swa_kv_pool)
                    if swa_l
                    else hicache_auto_size._pool_bytes(d_sp) * p_sp.size // d_sp.size
                )
                self.assertEqual(hicache_auto_size._draft_bytes(p_sp, d_sp), exp)

        t_pool, t_alloc = _make_target()
        d_pool = _attach_draft(t_alloc, full_layers=(0,), swa_layers=())
        r1 = SimpleNamespace(
            token_to_kv_pool=d_pool,
            model_config=SimpleNamespace(
                num_nextn_predict_layers=1,
                hf_config=SimpleNamespace(architectures=["EagleLlamaForCausalLM"]),
            ),
        )
        self.assertTrue(_can_pack_hicache_mtp(SpeculativeAlgorithm.EAGLE, (r1,)))
        worker = object.__new__(BaseSpecWorker)
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                token_to_kv_pool=t_pool,
                mtp_draft_device_pools=(),
                spec_algorithm=SpeculativeAlgorithm.EAGLE,
            )
        )
        worker._draft_worker = SimpleNamespace(draft_runners=[r1])
        mem_on = SimpleNamespace(
            enable_hierarchical_cache=True, enable_unified_cache_external_linker=False
        )
        with patch("sglang.srt.runtime_context.get_memory", return_value=mem_on):
            plan = worker._build_hicache_draft_plan()
            self.assertEqual(
                (plan.mode, plan.device_pools), (HiCacheDraftMode.SIDECAR, (d_pool,))
            )
            worker._draft_worker = SimpleNamespace(draft_runners=[r1, r1])
            with self.assertRaisesRegex(NotImplementedError, "Multi-runner MTP"):
                worker._build_hicache_draft_plan()

        swa_t, swa_a = _make_target()
        swa_d = _attach_draft(swa_a, full_layers=(), swa_layers=(0, 1))
        for tp, dp, exp_names in (
            (swa_t, swa_d, []),
            (t_pool, d_pool, [PoolName.DRAFT]),
        ):
            kv_h, _ = _build_pair(tp)
            entry = hybrid_pool_assembler.build_pool_entry(
                name=PoolName.KV,
                host_pool=kv_h,
                device_pool=tp,
                layer_mapping={0: 0, 3: 1},
                transfer_layer_id_max=7,
                is_anchor=True,
            )
            tree = SimpleNamespace(
                cache_controller=SimpleNamespace(
                    mem_pool_host=HostPoolGroup([entry]), page_size=4
                )
            )
            specs, _ = hybrid_pool_assembler.build_hicache_draft_sidecars(
                draft_device_pools=(dp,), tree_cache=tree
            )
            self.assertEqual([s.pool_name for s in specs], exp_names)

        bm_t, bm_a = _make_target()
        bm_d = _attach_draft(
            bm_a,
            full_layers=(),
            swa_layers=(0, 1),
            pool_cls=TpuBlockMajorMHATokenToKVPool,
        )
        self.assertEqual(bm_d.get_contiguous_buf_infos(), ([], [], []))
        self.assertEqual(len(get_raiden_kv_caches(bm_t, draft_pool=bm_d).draft), 1)
        with self.assertRaisesRegex(ValueError, "Mixed full and sliding-window"):
            _attach_draft(_make_target()[1], full_layers=(0,), swa_layers=(1,))


class _CopyingRaidenEngine:
    registry: dict[str, _CopyingRaidenEngine] = {}

    def __init__(
        self,
        *,
        kv_caches,
        local_control_port: int,
        max_blocks: int,
        num_slots: int,
        **kw,
    ):
        self.kv_caches = list(kv_caches)
        self.port = local_control_port
        self.endpoint = f"127.0.0.1:{local_control_port}"
        self.registered: dict[tuple[str, int], list[int]] = {}
        self._sent: list[str] = []
        self._recv: list[str] = []
        _CopyingRaidenEngine.registry[self.endpoint] = self

    def register_read(self, req_id: str, uuid: int, block_ids: list[int]) -> bool:
        self.registered[(req_id, uuid)] = list(block_ids)
        return True

    def start_read(self, req_id, uuid, endpoint, remote, local) -> None:
        peer = _CopyingRaidenEngine.registry[endpoint]
        assert len(remote) == len(local) and set(remote) <= set(
            peer.registered[(req_id, uuid)]
        )
        for src, dst in zip(peer.kv_caches, self.kv_caches, strict=True):
            dst[local] = src[remote]
        self._recv.append(req_id)
        peer._sent.append(req_id)

    def poll_stats_with_details(self):
        stat = lambda r: SimpleNamespace(
            req_id=r,
            completed_ns=10**9,
            elapsed_ns=None,
            peer=None,
            code=None,
            reason=None,
            phase=None,
        )
        out = SimpleNamespace(
            sent=[stat(r) for r in self._sent],
            received=[stat(r) for r in self._recv],
            failed=[],
        )
        self._sent.clear()
        self._recv.clear()
        return out


class TestUnifiedSpecRaidenPD(RaidenRendezvousTestCase):
    def setUp(self) -> None:
        super().setUp()
        reset_context()
        publish(ServerArgs(model_path="dummy"), role="scheduler")
        self.addCleanup(reset_context)
        _CopyingRaidenEngine.registry.clear()
        p = patch(
            "sglang_tpu.kv_pool._seq_along_lane_head_dim",
            side_effect=lambda *, head_dim, kv_packing: (
                ((head_dim + 8 * kv_packing - 1) // (8 * kv_packing)) * (8 * kv_packing)
            ),
        )
        p.start()
        self.addCleanup(p.stop)

    def _pump(self, *mgrs) -> None:
        progressed = True
        while progressed:
            progressed = False
            for m in mgrs:
                frames = self.inboxes[m.endpoint].frames
                while frames:
                    m._dispatch(frames.pop(0))
                    progressed = True

    def _setup_mgr(
        self, mgr, target_pool, draft_pool, *, local_rank: int, chunked: bool
    ):
        mgr.server_args.nnodes, mgr.is_dummy_cp_rank = 1, False
        mgr._engine = mgr._swa_engine = mgr._state_engine = mgr._draft_engine = None
        mgr._local_rank = local_rank
        mgr._control_port = 41000 + PORTS_PER_RANK * local_rank
        mgr.local_ip = "127.0.0.1"
        mgr.kv_args.state_types = [StateType.SWA]
        mgr.kv_args.kv_item_lens = target_pool.kv_pool.get_contiguous_buf_infos()[2]
        mgr.kv_item_lens_sum = sum(mgr.kv_args.kv_item_lens)
        mgr._max_request_blocks, mgr._swa_max_request_blocks = (lambda: 16), (lambda: 4)
        mgr._listener_start_read = False
        with patch.dict(sys.modules, _fake_tpu_sync(_CopyingRaidenEngine)):
            mgr.set_kv_caches(get_raiden_kv_caches(target_pool, draft_pool=draft_pool))
        mgr.chunked_transfer = chunked
        return mgr

    def _run_pd_case(
        self,
        room: int,
        t_full,
        t_swa,
        d_full,
        d_swa,
        pool_cls,
        chunked: bool,
        cover_all: bool,
        dec_nf: int,
        dec_ns: int,
    ) -> None:
        _CopyingRaidenEngine.registry.clear()
        p_t, p_a = _make_target(
            size=128, full_layers=t_full, swa_layers=t_swa, pool_cls=pool_cls
        )
        p_d = _attach_draft(
            p_a, full_layers=d_full, swa_layers=d_swa, pool_cls=pool_cls
        )
        d_t, d_a = _make_target(
            size=128, full_layers=t_full, swa_layers=t_swa, pool_cls=pool_cls
        )
        d_d = _attach_draft(
            d_a, full_layers=d_full, swa_layers=d_swa, pool_cls=pool_cls
        )

        pre = self._setup_mgr(self._prefill(), p_t, p_d, local_rank=0, chunked=chunked)
        dec = self._setup_mgr(self._decode(), d_t, d_d, local_rank=1, chunked=chunked)
        self.assertEqual(pre.kv_layout, UNIFIED_SWA_KV_LAYOUT)
        self.assertEqual(pre._draft_engine.port, pre._control_port + DRAFT_PORT_OFFSET)

        p_ft, p_st = p_a.full_attn_allocator.alloc(16), p_a.swa_attn_allocator.alloc(12)
        p_a.set_full_to_swa_mapping(p_ft[-12:], p_st)
        p_fp, p_sp = (p_ft[::4] // 4).tolist(), (p_st[::4] // 4).tolist()
        p_se = p_t.expand_swa_pages(p_sp)

        _ = d_a.full_attn_allocator.alloc(8)
        d_ft = (
            d_a.full_attn_allocator.alloc(dec_nf * 4)
            if dec_nf
            else torch.empty(0, dtype=torch.int64)
        )
        d_st = (
            d_a.swa_attn_allocator.alloc(dec_ns * 4)
            if dec_ns
            else torch.empty(0, dtype=torch.int64)
        )
        if dec_nf and dec_ns:
            k = min(len(d_ft), len(d_st))
            d_a.set_full_to_swa_mapping(d_ft[-k:], d_st[-k:])
        d_fp, d_sp = (
            ((d_ft[::4] // 4).tolist() if dec_nf else []),
            ((d_st[::4] // 4).tolist() if dec_ns else []),
        )

        _fill_pages(p_t.kv_pool.page_buffers(), p_fp)
        _fill_pages(p_t.kv_pool.page_buffers(), p_se)
        p_dsub, d_dsub = (
            (p_d.swa_kv_pool if d_swa else p_d),
            (d_d.swa_kv_pool if d_swa else d_d),
        )
        p_dp = _translate_dswa(p_a, p_sp) if d_swa else p_fp
        _fill_pages(page_major_tensors(p_dsub), p_dp)

        pre.request_status[room] = dec.request_status[room] = KVPoll.Bootstrapping
        recv_obj = object.__new__(RaidenKVReceiver)
        recv_obj.kv_mgr, recv_obj.bootstrap_room, recv_obj.conclude_state = (
            dec,
            room,
            None,
        )
        recv_obj.bootstrap_infos = self._bootstrap_info(pre)
        recv_obj.send_metadata(
            np.asarray(d_fp, dtype=np.int32),
            aux_index=None,
            state_indices=[np.asarray(d_sp, dtype=np.int32)],
        )
        self._pump(pre, dec)

        send_obj = object.__new__(RaidenKVSender)
        send_obj.kv_mgr, send_obj.bootstrap_room = pre, room
        send_obj.aux_index = send_obj._transfer_start_time = send_obj.conclude_state = (
            None
        )
        send_obj._transfer_metric = KVTransferMetric()
        send_obj._transfer_num_kv_indices = send_obj._transfer_num_state_indices = 0
        send_obj._started = False
        pre.get_kv_replica_factor = lambda: 1

        swa_arg = [np.asarray(p_sp, dtype=np.int32)]
        if chunked and cover_all:
            (
                pre._prefill.announce_pages,
                send_obj.num_kv_indices,
                send_obj._last_chunk,
            ) = 1, 2, False
            send_obj.send(np.asarray(p_fp[:2], dtype=np.int32), state_indices=None)
            send_obj._last_chunk = True
            send_obj.send(np.asarray([], dtype=np.int32), state_indices=swa_arg)
        elif chunked:
            (
                pre._prefill.announce_pages,
                send_obj.num_kv_indices,
                send_obj._last_chunk,
            ) = 2, len(p_fp), False
            send_obj.send(np.asarray(p_fp[:2], dtype=np.int32), state_indices=None)
            self._pump(pre, dec)
            dec.maybe_start_pull(room)
            pre._drain_engine()
            dec._drain_engine()
            send_obj._last_chunk = True
            send_obj.send(np.asarray(p_fp[2:], dtype=np.int32), state_indices=swa_arg)
        else:
            send_obj.num_kv_indices, send_obj._last_chunk = len(p_fp), True
            send_obj.send(np.asarray(p_fp, dtype=np.int32), state_indices=swa_arg)

        for _ in range(6):
            self._pump(pre, dec)
            dec.maybe_start_pull(room)
            pre._drain_engine()
            dec._drain_engine()
            if pre.is_send_done(room) and dec.is_recv_done(room):
                break
        self.assertTrue(pre.is_send_done(room) and dec.is_recv_done(room))
        self.assertFalse(pre.sender_has_active_io(room) or dec.room_has_active_io(room))

        if dec_nf:
            src_f = p_fp[:dec_nf] if (chunked and cover_all) else p_fp[-dec_nf:]
            for sb, rb in zip(p_t.kv_pool.page_buffers(), d_t.kv_pool.page_buffers()):
                self.assertTrue(torch.equal(rb[d_fp], sb[src_f]))
            if d_full:
                for sb, rb in zip(
                    page_major_tensors(p_dsub), page_major_tensors(d_dsub)
                ):
                    self.assertTrue(torch.equal(rb[d_fp], sb[src_f]))
        if dec_ns:
            m_t = p_t.layout.num_swa_groups
            for sb, rb in zip(p_t.kv_pool.page_buffers(), d_t.kv_pool.page_buffers()):
                self.assertTrue(
                    torch.equal(
                        rb[d_t.expand_swa_pages(d_sp)], sb[p_se[-dec_ns * m_t :]]
                    )
                )
            if d_swa:
                dst_dp = _translate_dswa(d_a, d_sp)
                for sb, rb in zip(
                    page_major_tensors(p_dsub), page_major_tensors(d_dsub)
                ):
                    self.assertTrue(torch.equal(rb[dst_dp], sb[p_dp[-dec_ns:]]))

        n_kv = 2 if (chunked and cover_all) else len(p_fp)
        self.assertEqual(
            send_obj.get_transfer_metric().transfer_total_bytes,
            n_kv * pre.kv_item_lens_sum + len(p_sp) * pre.state_item_lens_sum,
        )
        pre.clear_room(room)
        dec.clear_room(room)

    def test_equal_tp_pd_transfer(self):
        self.assertEqual(PORTS_PER_RANK, 8)
        topologies = (
            ("fa_m3", (0, 3), (1, 2, 4, 5, 6), (0, 1), (), TpuFusedMHATokenToKVPool),
            (
                "swa_m3",
                (0, 3),
                (1, 2, 4, 5, 6),
                (),
                (0, 1, 2),
                TpuFusedMHATokenToKVPool,
            ),
            ("swa_m1", (0, 3), (1, 2), (), (0, 1), TpuFusedMHATokenToKVPool),
            ("swa_seq", (0, 3), (1, 2, 4, 5, 6), (), (0, 1), TpuSeqAlongLaneKVPool),
        )
        modes = (
            ("chunked", True, False, 4, 2),
            ("chunked_cover_all", True, True, 2, 2),
            ("whole_pull", False, False, 4, 2),
            ("partial_hit", True, False, 2, 2),
            ("swa_only_hit", True, False, 0, 2),
            ("full_hit", True, False, 0, 0),
        )
        room = 9000
        for t_name, tf, ts, df, ds, cls in topologies:
            for m_name, chunked, cover_all, dnf, dns in modes:
                with self.subTest(topology=t_name, mode=m_name):
                    self._run_pd_case(
                        room, tf, ts, df, ds, cls, chunked, cover_all, dnf, dns
                    )
                    room += 1

    def test_asymmetric_tp_raises_not_implemented(self):
        for df, ds in (((0,), ()), ((), (0,))):
            t_p, t_a = _make_target()
            d_p = _attach_draft(t_a, full_layers=df, swa_layers=ds)
            pre = self._setup_mgr(
                self._prefill(attn_tp_rank=0, attn_tp_size=4),
                t_p,
                d_p,
                local_rank=0,
                chunked=True,
            )
            dec = self._setup_mgr(
                self._decode(attn_tp_rank=0, attn_tp_size=2),
                t_p,
                d_p,
                local_rank=1,
                chunked=True,
            )
            peers = [
                {"rank_ip": "127.0.0.1", "rank_port": 40001},
                {"rank_ip": "127.0.0.1", "rank_port": 40002},
            ]
            with self.assertRaisesRegex(NotImplementedError, "equal attention TP"):
                dec.register_decode_request(9901, [1, 2], [4, 5], None, peers)
            req = ReshardReq(
                room=9902,
                decode_ip="127.0.0.1",
                decode_port=40010,
                push_endpoint="127.0.0.1:41010",
                decode_blocks=[1],
                dst_attn_tp=2,
                decode_swa_blocks=[],
                swa_push_endpoint="",
            )
            with self.assertRaisesRegex(NotImplementedError, "equal attention TP"):
                pre._on_reshard_req(req.to_frames())


if __name__ == "__main__":
    unittest.main()
