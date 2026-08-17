#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Tests for scripts/isa_resource_table.py.

The ISA text is generated here rather than checked in, so the file states exactly
which parts of LLVM's output the parser depends on: the metadata indentation, the
`.set <kernel>.*` symbols, the `.size` terminator, and the mnemonic spelling.

Targets differ along three axes the parser has to survive, so every assertion runs
over one shape from each side of them:

* whether the target has an accumulator register file — LLVM emits `.agpr_count`
  only when `STM.hasMAIInsts()`, so an MFMA-capable target has the field and a
  target without MFMA does not;
* how LDS access is spelled — `ds_read`/`ds_write` through gfx10, renamed to
  `ds_load`/`ds_store` from gfx11 on;
* how the target ID spells the triple's environment field —
  `amdgcn-amd-amdhsa--gfx942` when it is empty, `amdgcn-amd-amdhsa-unknown-gfx942`
  when it is `unknown`. This repo's own dumps use the second, and an earlier
  version of this file exercised only the first, so the tool shipped unable to
  identify the architecture of any dump it would actually be pointed at.

The first two are ISA properties, not product families, and they do not track the
CDNA / RDNA marketing split: gfx1250 is a CDNA-generation part that LLVM places in
the GFX12 family (wave32, no `FeatureMAIInsts`), so it takes the second value on
both. An earlier version of this tool was tested only against a gfx9-shaped sample
and so stayed green while rejecting every kernel on gfx12xx. The third is a
property of how the triple reached the backend rather than of the target itself,
so it is pinned per shape only to keep both spellings under test.

The generated shape was checked against real dumps: its metadata keys are a subset
of the keys LLVM emits, in the same sorted order, and every structural marker the
parser keys off (`.amdgcn_target`, the kernel label, `.Lfunc_endN`, `.size`, the
`.set` block, the `  - `/4-space/6-space indent levels) appears as it does there.
The keys left out are ones the parser never reads.
"""

import pytest

from scripts import isa_resource_table as irt
from scripts.isa_resource_table import VALUE, parse_isa

pytestmark = [pytest.mark.l0_backend_agnostic]

LDS_READS = 3


def make_isa(arch, kernel, *, mfma, ds_load_spelling, num_vgpr, num_agpr, vgpr_count, env):
    """One kernel's final ISA, in the shape LLVM emits for the given target.

    `mfma` selects whether the target has an accumulator register file, which is
    what decides both the `.agpr_count` metadata field and the matrix mnemonic.
    `ds_load_spelling` selects the gfx11+ LDS naming. `env` is the triple's
    environment field, which `AMDGPUTargetID::toString()` writes out verbatim
    between the OS and the processor. Everything else is common.
    """
    target_id = f"amdgcn-amd-amdhsa-{env}-{arch}"
    read = "ds_load_b128" if ds_load_spelling else "ds_read_b64"
    write = "ds_store_b32" if ds_load_spelling else "ds_write_b32"
    # Operands are v-registers on both sides: gfx90a+ unified the register file, and
    # a target without an accumulator file has no `a[...]` to write to at all.
    matrix = (
        "v_mfma_f32_16x16x16_f16 v[0:3], v4, v5, v[0:3]"
        if mfma
        else "v_wmma_f32_16x16x32_bf16 v[0:7], v[8:15], v[16:23], 0"
    )
    lines = [
        f'\t.amdgcn_target "{target_id}"',
        "\t.text",
        f"{kernel}:                               ; @{kernel}",
        *[f"\t{read} v[0:1], v2"] * LDS_READS,
        f"\t{write} v3, v4",
        f"\t{matrix}",
        "\ts_endpgm",
        # The parser must skip the descriptor block and bound the body at the
        # `.size` terminator, not at s_endpgm.
        '\t.section\t.rodata,"a",@progbits',
        f"\t.amdhsa_kernel {kernel}",
        f"\t\t.amdhsa_next_free_vgpr {num_vgpr}",
        "\t.end_amdhsa_kernel",
        "\t.text",
        ".Lfunc_end0:",
        f"\t.size\t{kernel}, .Lfunc_end0-{kernel}",
        # Register counts come from these symbols, which every target emits.
        f"\t.set {kernel}.num_vgpr, {num_vgpr}",
        f"\t.set {kernel}.num_agpr, {num_agpr}",
        f"\t.set {kernel}.numbered_sgpr, 53",
        f"\t.set {kernel}.private_seg_size, 0",
        "\t.set amdgpu.max_num_vgpr, 0",  # module-level decoy, must not bind to a kernel
        "\t.amdgpu_metadata",
        "amdhsa.kernels:",
    ]
    # Metadata indentation is a contract: "  - " opens an entry, kernel keys sit
    # at column 4, and argument keys are deeper so they must not be mistaken for
    # kernel keys. Keys are emitted in sorted order, as LLVM does.
    lines += [f"  - .agpr_count:     {num_agpr}", "    .args:"] if mfma else ["  - .args:"]
    lines += [
        "      - .offset:         0",
        "        .size:           8",
        "        .value_kind:     global_buffer",
        "    .group_segment_fixed_size: 39936",
        f"    .name:           {kernel}",
        "    .private_segment_fixed_size: 0",
        "    .sgpr_count:     59",
        "    .sgpr_spill_count: 0",
        f"    .symbol:         {kernel}.kd",
        f"    .vgpr_count:     {vgpr_count}",
        "    .vgpr_spill_count: 0",
        f"amdhsa.target:   {target_id}",
        ".end_amdgpu_metadata",
    ]
    return "\n".join(lines) + "\n"


TARGETS = {
    # Accumulators in use, gfx9 LDS spelling. `.vgpr_count` is the arch+acc total,
    # which is why it, and not its split, is the VGPR regression trigger.
    "gfx942": dict(
        arch="gfx942",
        kernel="gemm_0",
        mfma=True,
        ds_load_spelling=False,
        num_vgpr=256,
        num_agpr=29,
        vgpr_count=285,
        env="unknown",
    ),
    # No accumulator register file, so no `.agpr_count` anywhere; gfx11+ LDS
    # spelling. The AGPR count must still come through, from the `.set` symbol.
    "gfx1250": dict(
        arch="gfx1250",
        kernel="fmha_0",
        mfma=False,
        ds_load_spelling=True,
        num_vgpr=942,
        num_agpr=0,
        vgpr_count=942,
        env="",
    ),
}


def dump_tree(root, spec, **overrides):
    return write_tree(root, make_isa(**{**spec, **overrides}))


def write_tree(root, text):
    """A one-directory dump tree holding exactly the given ISA text."""
    (root / "k").mkdir(parents=True, exist_ok=True)
    (root / "k" / "21_final_isa.s").write_text(text)
    return root


def with_second_entry(text, mutate):
    """`text` with a second kernel entry spliced into the metadata list.

    A single-entry file cannot express "one entry was dropped": losing the only
    kernel already raises, which hides the case where a healthy kernel keeps a
    damaged sibling company and the snapshot still claims to be complete.
    """
    head, target = text.split("\namdhsa.target:", 1)
    entry = head.split("amdhsa.kernels:\n", 1)[1]
    return head + "\n" + mutate(entry).rstrip("\n") + "\namdhsa.target:" + target


@pytest.mark.parametrize("target", sorted(TARGETS), ids=sorted(TARGETS))
def test_resources_are_read_and_diffed_on_both_target_shapes(tmp_path, target):
    spec = TARGETS[target]
    before = dump_tree(tmp_path / "before", spec)

    records, file_problems = parse_isa(before / "k" / "21_final_isa.s")
    assert file_problems == ()
    (record,) = records.values()
    assert record.name == spec["kernel"], "identity comes from the kernel, not its first argument"

    # Nothing may be unreadable on a healthy dump from either shape. The AGPR count
    # in particular must come from the `.set` symbol, which every target emits,
    # rather than from `.agpr_count`, which only MFMA-capable targets carry.
    assert record.unparsed_keys == ()
    assert record.metrics["agpr"] == irt.Cell.of(spec["num_agpr"])
    assert record.metrics["vgpr"] == irt.Cell.of(spec["vgpr_count"])
    assert record.metrics["arch_vgpr"] == irt.Cell.of(spec["num_vgpr"])

    # LDS traffic is counted under whichever spelling this target uses; a parser
    # that knew only one of them would silently report 0 on the other.
    assert record.metrics["lds_read"] == irt.Cell.of(LDS_READS)
    assert record.metrics["lds_write"].state == VALUE

    # An unchanged pair is clean; a higher VGPR total is a regression; a missing
    # input is untrustworthy and must never be reported as either of the first two.
    assert irt.main(["diff", str(before), str(dump_tree(tmp_path / "same", spec))]) == 0
    worse = dump_tree(tmp_path / "worse", spec, vgpr_count=spec["vgpr_count"] + 8)
    assert irt.main(["diff", str(before), str(worse)]) == 1
    assert irt.main(["diff", str(before), str(tmp_path / "nonexistent")]) == 2


def test_target_id_is_parsed_under_both_environment_spellings():
    """The processor is the last `-` separated field, not whatever follows a `--`.

    `AMDGPUTargetID::toString()` writes the triple's environment out verbatim, so an
    empty environment gives `...amdhsa--gfx942` and an `unknown` one gives
    `...amdhsa-unknown-gfx942`. Recognizing only the first leaves `processor` unset on
    every dump this repo actually produces, which disables the architecture-mismatch
    guard and the scratch-versus-buffer spill classification at the same time, with
    nothing but a warning to show for it.
    """
    for target_id in (
        "amdgcn-amd-amdhsa--gfx942",
        "amdgcn-amd-amdhsa-unknown-gfx942",
        "amdgcn-amd-amdhsa-unknown-gfx942:sramecc+:xnack-",
    ):
        arch = irt.parse_target_id(target_id)
        assert arch.processor == "gfx942", target_id
        assert arch.gen == 9, target_id
        assert arch.spills_via_scratch() is True, target_id

    # A target ID with no concrete processor must stay unidentified rather than take
    # its last field on faith: `gfx11-generic` really does end in `generic`.
    assert irt.parse_target_id("amdgcn-amd-amdhsa-unknown-gfx11-generic").processor is None
    assert irt.parse_target_id("amdgcn-amd-amdhsa").processor is None


def test_architecture_mismatch_is_never_reported_as_comparable(tmp_path):
    """Two different targets are exit 2, whether or not the tool can name them.

    Register counts from different architectures are not comparable quantities, and
    an unidentified processor is not evidence that the two sides match -- comparing
    them anyway is how a gfx942-against-gfx950 run printed RESULT: OK.
    """
    spec = TARGETS["gfx942"]
    before = dump_tree(tmp_path / "before", spec)
    assert irt.main(["diff", str(before), str(dump_tree(tmp_path / "same", spec))]) == 0
    assert irt.main(["diff", str(before), str(dump_tree(tmp_path / "gfx950", spec, arch="gfx950"))]) == 2

    # A target the tool cannot name blocks on its own: without a processor it can
    # decide neither that the two sides match nor whether the scratch metrics apply.
    # `gfx11-generic` really does end in `generic`, so this is reachable, and the fix
    # if it ever matters is to teach `RE_PROC` about it, not to loosen the verdict.
    generic = dump_tree(tmp_path / "generic11", spec, arch="gfx11-generic")
    assert irt.parse_isa(generic / "k" / "21_final_isa.s")[0].popitem()[1].arch.known is False
    assert irt.main(["diff", str(generic), str(dump_tree(tmp_path / "generic11b", spec, arch="gfx11-generic"))]) == 2
    assert irt.main(["diff", str(generic), str(dump_tree(tmp_path / "generic12", spec, arch="gfx12-generic"))]) == 2


def test_an_unreadable_final_isa_blocks_the_comparison(tmp_path):
    """A discovered dump file the parser cannot read must not degrade to a warning.

    Skipping it quietly leaves a comparison that is partial but still prints a
    full-coverage count line and RESULT: OK, which is the single answer the exit-2
    verdict exists to prevent.
    """
    spec = TARGETS["gfx942"]
    before, after = tmp_path / "before", tmp_path / "after"
    for side in (before, after):
        dump_tree(side, spec)
        (side / "truncated").mkdir(parents=True)
        (side / "truncated" / "21_final_isa.s").write_text("")

    # The healthy half compares cleanly on its own, so the verdict below is
    # attributable to the unreadable file and to nothing else in the tree.
    assert irt.main(["diff", str(dump_tree(tmp_path / "a", spec)), str(dump_tree(tmp_path / "b", spec))]) == 0
    assert irt.main(["diff", str(before), str(after)]) == 2


def test_target_features_are_part_of_comparability(tmp_path):
    """Same processor is not the same target, and the same target has two spellings.

    `xnack` and `sramecc` change code generation and register allocation, so a diff
    taken across them reports the build flags rather than the change under test. The
    triple's environment field, by contrast, is empty or `unknown` for one and the
    same target, so it must not read as a difference.
    """
    spec = TARGETS["gfx942"]
    base = make_isa(**spec)
    feature = lambda suffix: write_tree(tmp_path / f"f{suffix}", base.replace("gfx942", f"gfx942:{suffix}"))

    # Environment spelling: same target, so still comparable.
    empty_env = write_tree(tmp_path / "empty_env", base.replace("amdhsa-unknown-gfx942", "amdhsa--gfx942"))
    assert irt.main(["diff", str(dump_tree(tmp_path / "plain", spec)), str(empty_env)]) == 0

    assert irt.main(["diff", str(feature("xnack+")), str(feature("xnack+"))]) == 0
    assert irt.main(["diff", str(feature("xnack+")), str(feature("xnack-"))]) == 2
    assert irt.main(["diff", str(feature("xnack+")), str(feature("sramecc+:xnack+"))]) == 2

    # Neither side declares a target at all: there is no evidence the two runs are the
    # same target, and matching absences are not evidence of a match.
    strip = lambda t: "".join(
        line for line in t.splitlines(keepends=True) if ".amdgcn_target" not in line and "amdhsa.target:" not in line
    )
    assert (
        irt.main(["diff", str(write_tree(tmp_path / "n1", strip(base))), str(write_tree(tmp_path / "n2", strip(base)))])
        == 2
    )


def test_a_negative_resource_count_is_never_reported_as_an_improvement(tmp_path):
    """Counts and byte sizes cannot go below zero, so a negative one means a bad dump.

    Accepting it is worse than dropping the kernel: the fall toward it is rendered as
    a resource win, which is the one verdict this tool must never invent.
    """
    spec = TARGETS["gfx942"]
    base = make_isa(**spec)
    before = dump_tree(tmp_path / "before", spec)
    negative = write_tree(tmp_path / "negative", base.replace(".vgpr_count:     285", ".vgpr_count:     -1"))
    assert irt.main(["diff", str(before), str(negative)]) == 2

    # Same for the `.set` symbols, which feed the informational split.
    symbol = write_tree(tmp_path / "symbol", base.replace("gemm_0.num_agpr, 29", "gemm_0.num_agpr, -29"))
    assert irt.main(["diff", str(before), str(symbol)]) == 2


def test_a_kernel_entry_without_or_with_a_duplicate_identity_is_a_problem(tmp_path):
    """An entry the tool cannot key must not vanish from the snapshot.

    Both mutations leave a healthy kernel behind, so the run still produces a table;
    what it must not do is call that table complete.
    """
    spec = TARGETS["gfx942"]
    base = make_isa(**spec)

    anonymous = with_second_entry(
        base,
        lambda e: e.replace("    .name:           gemm_0\n", "").replace("    .symbol:         gemm_0.kd\n", ""),
    )
    snapshot = irt.collect(write_tree(tmp_path / "anonymous", anonymous))
    assert len(snapshot) == 1 and not snapshot.trustworthy

    # A second entry under a name already seen would otherwise overwrite the first,
    # so a stale duplicate could hide a changed counter behind an unchanged one.
    duplicated = with_second_entry(base, lambda e: e.replace(".vgpr_count:     285", ".vgpr_count:     999"))
    snapshot = irt.collect(write_tree(tmp_path / "duplicated", duplicated))
    assert snapshot.kernels["k::gemm_0"].metrics["vgpr"] == irt.Cell.of(spec["vgpr_count"])
    assert not snapshot.trustworthy


def test_an_undecodable_byte_makes_the_instruction_counts_untrustworthy(tmp_path):
    """A damaged byte can erase a mnemonic, and an erased mnemonic is not a zero.

    Lenient decoding is still the right call -- a traceback would report worse than
    exit 2 does -- but the replacement has to be admitted rather than counted.
    """
    spec = TARGETS["gfx942"]
    path = dump_tree(tmp_path / "corrupt", spec) / "k" / "21_final_isa.s"
    raw = path.read_bytes()
    cut = raw.index(b"ds_read_b64")
    path.write_bytes(raw[: cut + 3] + b"\xff" + raw[cut + 4 :])

    records, file_problems = parse_isa(path)
    assert file_problems, "an undecodable byte has to be reported, not absorbed"
    assert records["gemm_0"].metrics["lds_read"] == irt.Cell.of(LDS_READS - 1), "the count really did change"
    assert irt.main(["diff", str(dump_tree(tmp_path / "clean", spec)), str(tmp_path / "corrupt")]) == 2
