#!/usr/bin/env python3
"""
Merge CloudFormation template parameter defaults with explicit Key=Value overrides.

Reads Parameters.*.Default from the template, applies overrides (later wins), validates
required parameters, and prints one Key=Value per line for aws cloudformation deploy.
"""

from __future__ import annotations

import sys


def _format_default(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return ""
    return str(value)


def _validate_ssm_parameter_path_types(params: dict, merged: dict[str, str]) -> None:
    """
    CloudFormation AWS::SSM::Parameter::Value<...> expects the SSM *parameter name* (path string).
    Users sometimes paste the JSON *value* from `aws ssm get-parameter` instead — reject that early.
    """
    for name, spec in params.items():
        if not isinstance(spec, dict):
            continue
        ptype = str(spec.get("Type", ""))
        if "AWS::SSM::Parameter::Value" not in ptype:
            continue
        val = merged.get(name, "").strip()
        if not val:
            continue
        if val.startswith("{"):
            print(
                f"cfn_merge_stack_parameters: parameter {name!r} must be the SSM parameter path "
                f"(for example /aws/service/ecs/optimized-ami/amazon-linux-2/recommended), "
                f"not the JSON document returned as the parameter value. "
                f"Use the SSM path from the template Default (e.g. /aws/service/ecs/optimized-ami/...), not JSON.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not val.startswith("/"):
            print(
                f"cfn_merge_stack_parameters: parameter {name!r} must be an SSM path starting with /, got {val!r}",
                file=sys.stderr,
            )
            sys.exit(1)


def _load_cfn_yaml(path: str):
    """Parse CloudFormation YAML (includes !Ref / !Sub / etc.) for the Parameters section."""
    import yaml

    class CfnSafeLoader(yaml.SafeLoader):
        pass

    def _construct_cfn_tag(loader, tag_suffix, node):
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        if isinstance(node, yaml.MappingNode):
            return loader.construct_mapping(node)
        return None

    CfnSafeLoader.add_multi_constructor("!", _construct_cfn_tag)

    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=CfnSafeLoader)


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "Usage: cfn_merge_stack_parameters.py <template.yaml> [Key=Value ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    template_path = sys.argv[1]
    override_args = sys.argv[2:]

    try:
        doc = _load_cfn_yaml(template_path)
    except ImportError:
        print(
            "cfn_merge_stack_parameters: PyYAML is required. "
            "Install with: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)
    except OSError as e:
        print(f"cfn_merge_stack_parameters: cannot read {template_path}: {e}", file=sys.stderr)
        sys.exit(1)

    params = doc.get("Parameters") or {}
    if not isinstance(params, dict):
        print("cfn_merge_stack_parameters: invalid template: Parameters must be a mapping", file=sys.stderr)
        sys.exit(1)

    merged: dict[str, str] = {}
    for name, spec in params.items():
        if not isinstance(spec, dict):
            continue
        if "Default" in spec:
            merged[name] = _format_default(spec["Default"])

    for raw in override_args:
        if "=" not in raw:
            print(f"cfn_merge_stack_parameters: invalid override (expected Key=Value): {raw!r}", file=sys.stderr)
            sys.exit(1)
        key, _, value = raw.partition("=")
        if key not in params:
            print(
                f"cfn_merge_stack_parameters: unknown parameter {key!r} (not in template {template_path})",
                file=sys.stderr,
            )
            sys.exit(1)
        merged[key] = value

    missing_required: list[str] = []
    for name, spec in params.items():
        if not isinstance(spec, dict):
            continue
        if "Default" in spec:
            continue
        if name not in merged:
            missing_required.append(name)

    if missing_required:
        print(
            "cfn_merge_stack_parameters: missing required parameter(s) (no Default in template): "
            + ", ".join(missing_required),
            file=sys.stderr,
        )
        sys.exit(1)

    _validate_ssm_parameter_path_types(params, merged)

    for key in sorted(merged.keys()):
        print(f"{key}={merged[key]}")


if __name__ == "__main__":
    main()
