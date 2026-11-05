"""Check cloud report metadata for the Phase 1 submission gate; never compute predictions."""
import argparse
import json
from pathlib import Path


def check_report(report,best_objective,expected_version=None,expected_device=None,expected_pop_weight=None):
    failures = []
    test = report.get("test",{})
    for field,expected in (("molecules",400),("fallback",0),("no_logits",0)):
        if test.get(field)!=expected:
            failures.append(f"test.{field} must equal {expected}")
    total = report.get("total_s")
    if not isinstance(total,(int,float)) or not 0<=total<10800:
        failures.append("total_s must be below 10800 seconds")
    selected = report.get("ranker",{}).get("selected",{}) if report.get("final_ranking")=="learned" else report.get("validation",{}).get("selected",{})
    objective = selected.get("objective")
    if not isinstance(objective,(int,float)) or not objective>=best_objective-0.02:
        failures.append("selected objective is missing or more than 0.02 below the best")
    config = report.get("config",{})
    if expected_version is not None and config.get("version")!=expected_version:
        failures.append("report experiment version does not match")
    if expected_device is not None and report.get("device")!=expected_device:
        failures.append("report device does not match")
    if expected_pop_weight is not None and config.get("pop_weight")!=expected_pop_weight:
        failures.append("report popularity weight does not match")
    return {"passed":not failures,"failures":failures,"selected":selected,"test":test,"total_s":total,
            "version":config.get("version"),"device":report.get("device")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--best-objective",type=float,required=True)
    parser.add_argument("--version")
    parser.add_argument("--device",choices=("cpu","gpu"))
    parser.add_argument("--pop-weight",type=float)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = check_report(json.loads(Path(args.report).read_text(encoding="utf-8")),args.best_objective,
                          args.version,args.device,args.pop_weight)
    result["report_path"] = str(Path(args.report).resolve())
    text = json.dumps(result,indent=2)+"\n"
    if args.output:
        Path(args.output).write_text(text,encoding="utf-8")
    print(text,end="")
    return 0 if result["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
