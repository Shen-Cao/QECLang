import QECLang.Checker

namespace QECLang

/-- A CSS-sector witness must also pair evenly with every opposite-sector check. -/
def checkCSSRaw (n : Nat) (same opposite : List (List Bool))
    (value dual : List Bool) : Bool :=
  match parseRows n opposite, parse n value with
  | some rows, some v =>
      if checkRows rows v then checkRaw n same value dual else false
  | _, _ => false

theorem checkCSSRaw_sound (n : Nat) (same opposite : List (List Bool))
    (value dual : List Bool) (h : checkCSSRaw n same opposite value dual = true) :
    ∃ (rows : List (Bits n)) (v : Bits n),
      parseRows n opposite = some rows ∧ parse n value = some v ∧
      Orthogonal rows v ∧ ¬ RawInSpan n same value ∧
      (∀ row ∈ same, row.length = n) ∧ dual.length = n := by
  cases hr : parseRows n opposite with
  | none => unfold checkCSSRaw at h; rw [hr] at h; cases h
  | some rows =>
    cases hv : parse n value with
    | none => unfold checkCSSRaw at h; rw [hr, hv] at h; cases h
    | some v =>
      cases hc : checkRows rows v with
      | false =>
        unfold checkCSSRaw at h
        rw [hr, hv] at h
        change (if checkRows rows v then checkRaw n same value dual else false) = true at h
        rw [hc] at h
        cases h
      | true =>
        have hraw : checkRaw n same value dual = true := by
          unfold checkCSSRaw at h
          rw [hr, hv] at h
          change (if checkRows rows v then checkRaw n same value dual else false) = true at h
          rw [hc] at h
          exact h
        have hl := checkRaw_lengths n same value dual hraw
        exact ⟨rows, v, rfl, rfl, checkRows_sound hc,
               checkRaw_not_in_span n same value dual hraw, hl.1, hl.2.2⟩

theorem checkCSSRaw_not_in_span (n : Nat) (same opposite : List (List Bool))
    (value dual : List Bool) (h : checkCSSRaw n same opposite value dual = true) :
    ¬ RawInSpan n same value := by
  obtain ⟨_, _, _, _, _, hn, _⟩ := checkCSSRaw_sound n same opposite value dual h
  exact hn

theorem checkCSSRaw_lengths (n : Nat) (same opposite : List (List Bool))
    (value dual : List Bool) (h : checkCSSRaw n same opposite value dual = true) :
    (∀ row ∈ same, row.length = n) ∧ (∀ row ∈ opposite, row.length = n) ∧
    value.length = n ∧ dual.length = n := by
  obtain ⟨_, _, hr, hv, _, _, hs, hd⟩ := checkCSSRaw_sound n same opposite value dual h
  exact ⟨hs, parseRows_lengths hr, parse_length hv, hd⟩

end QECLang
