import QECLang.Fixtures

def main : IO UInt32 := do
  let mut passed := true
  for (name, result) in QECLang.Fixtures.cases do
    IO.println s!"{if result then "PASS" else "FAIL"}: {name}"
    passed := passed && result
  return if passed then 0 else 1
