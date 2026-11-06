import QECLang.CssChecker

namespace QECLang.Fixtures

def steaneRows : List (List Bool) :=
  [[true, false, true, false, true, false, true],
   [false, true, true, false, false, true, true],
   [false, false, false, true, true, true, true]]

def logical : List Bool := [true, true, true, false, false, false, false]
def badDual : List Bool := [true, false, false, false, false, false, false]
def stabilizerRow : List Bool := [true, false, true, false, true, false, true]

theorem steane_accepts : checkRaw 7 steaneRows logical logical = true := by decide

theorem steane_nonmembership : ¬ RawInSpan 7 steaneRows logical :=
  checkRaw_not_in_span 7 steaneRows logical logical steane_accepts

theorem rejects_bad_dual : checkRaw 7 steaneRows logical badDual = false := by decide

theorem rejects_stabilizer : checkRaw 7 steaneRows stabilizerRow logical = false := by decide

theorem rejects_wrong_length : checkRaw 7 steaneRows [true] logical = false := by decide

theorem rejects_wrong_row_length :
    checkRaw 7 ([true] :: steaneRows) logical logical = false := by decide

theorem rejects_zero_width : checkRaw 0 [] [] [] = false := by decide

theorem css_steane_accepts :
    checkCSSRaw 7 steaneRows steaneRows logical logical = true := by decide

theorem noncentralizer_separates : checkRaw 7 steaneRows badDual logical = true := by decide

theorem css_rejects_noncentralizer :
    checkCSSRaw 7 steaneRows steaneRows badDual logical = false := by decide

theorem css_rejects_opposite_length :
    checkCSSRaw 7 steaneRows [[true]] logical logical = false := by decide

def cases : List (String × Bool) :=
  [("Steane logical witness", checkRaw 7 steaneRows logical logical),
   ("nonorthogonal dual rejected", !checkRaw 7 steaneRows logical badDual),
   ("stabilizer row rejected", !checkRaw 7 steaneRows stabilizerRow logical),
   ("candidate length rejected", !checkRaw 7 steaneRows [true] logical),
   ("row length rejected", !checkRaw 7 ([true] :: steaneRows) logical logical),
   ("zero width rejected", !checkRaw 0 [] [] []),
   ("CSS Steane witness", checkCSSRaw 7 steaneRows steaneRows logical logical),
   ("CSS noncentralizer rejected", !checkCSSRaw 7 steaneRows steaneRows badDual logical),
   ("CSS opposite width rejected", !checkCSSRaw 7 steaneRows [[true]] logical logical)]

end QECLang.Fixtures
