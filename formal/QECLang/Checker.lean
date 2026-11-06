import QECLang.Parity

namespace QECLang

/-- Reject a raw vector unless its length is exactly the declared width. -/
def parse : (n : Nat) → List Bool → Option (Bits n)
  | 0, [] => some .nil
  | n + 1, bit :: tail => do
      let rest ← parse n tail
      pure (.cons bit rest)
  | _, _ => none

def parseRows (n : Nat) : List (List Bool) → Option (List (Bits n))
  | [] => some []
  | row :: rows => do
      let value ← parse n row
      let rest ← parseRows n rows
      pure (value :: rest)

theorem parse_length {n : Nat} {raw : List Bool} {value : Bits n}
    (h : parse n raw = some value) : raw.length = n := by
  induction n generalizing raw with
  | zero =>
    cases raw with
    | nil => rfl
    | cons _ _ => cases h
  | succ n ih =>
    cases raw with
    | nil => cases h
    | cons bit tail =>
      cases ht : parse n tail with
      | none => unfold parse at h; rw [ht] at h; cases h
      | some rest => exact congrArg Nat.succ (ih ht)

theorem parseRows_lengths {n : Nat} {raw : List (List Bool)} {rows : List (Bits n)}
    (h : parseRows n raw = some rows) : ∀ row ∈ raw, row.length = n := by
  induction raw generalizing rows with
  | nil => intro row hr; cases hr
  | cons row tail ih =>
    cases hr : parse n row with
    | none => unfold parseRows at h; rw [hr] at h; cases h
    | some value =>
      cases ht : parseRows n tail with
      | none => unfold parseRows at h; rw [hr, ht] at h; cases h
      | some rest =>
        intro candidate hc
        cases hc with
        | head => exact parse_length hr
        | tail _ member => exact ih ht candidate member

/-- Check orthogonality using executable Boolean parity. -/
def checkRows (rows : List (Bits n)) (dual : Bits n) : Bool :=
  match rows with
  | [] => true
  | row :: rest => if dot row dual then false else checkRows rest dual

theorem checkRows_sound {rows : List (Bits n)} {dual : Bits n}
    (h : checkRows rows dual = true) : Orthogonal rows dual := by
  induction rows with
  | nil => intro row hr; cases hr
  | cons row rest ih =>
    cases hp : dot row dual with
    | false =>
      have ht : checkRows rest dual = true := by
        unfold checkRows at h
        rw [hp] at h
        exact h
      intro candidate hc
      cases hc with
      | head => exact hp
      | tail _ member => exact ih ht candidate member
    | true => unfold checkRows at h; rw [hp] at h; cases h

/-- Check a separating dual vector against every row and the candidate. -/
def check (rows : List (Bits n)) (value dual : Bits n) : Bool :=
  if checkRows rows dual then dot value dual else false

theorem check_properties {rows : List (Bits n)} {value dual : Bits n}
    (h : check rows value dual = true) :
    Orthogonal rows dual ∧ dot value dual = true := by
  cases hr : checkRows rows dual with
  | false => unfold check at h; rw [hr] at h; cases h
  | true =>
    unfold check at h
    rw [hr] at h
    exact ⟨checkRows_sound hr, h⟩

theorem check_sound {rows : List (Bits n)} {value dual : Bits n}
    (h : check rows value dual = true) : ¬ InSpan rows value := by
  exact odd_pairing_not_in_span (check_properties h).1 (check_properties h).2

/-- Executable boundary for concrete lists; all lengths are checked by parse. -/
def checkRaw (n : Nat) (rows : List (List Bool)) (value dual : List Bool) : Bool :=
  match parseRows n rows, parse n value, parse n dual with
  | some rs, some v, some d => check rs v d
  | _, _, _ => false

/-- A true raw check certifies a well-shaped interpretation and nonmembership. -/
theorem checkRaw_sound (n : Nat) (rows : List (List Bool)) (value dual : List Bool)
    (h : checkRaw n rows value dual = true) :
    ∃ (rs : List (Bits n)) (v d : Bits n),
      parseRows n rows = some rs ∧ parse n value = some v ∧ parse n dual = some d ∧
      Orthogonal rs d ∧ dot v d = true ∧ ¬ InSpan rs v := by
  cases hr : parseRows n rows with
  | none => unfold checkRaw at h; rw [hr] at h; cases h
  | some rs =>
    cases hv : parse n value with
    | none => unfold checkRaw at h; rw [hr, hv] at h; cases h
    | some v =>
      cases hd : parse n dual with
      | none => unfold checkRaw at h; rw [hr, hv, hd] at h; cases h
      | some d =>
        have hc : check rs v d = true := by
          unfold checkRaw at h
          rw [hr, hv, hd] at h
          exact h
        exact ⟨rs, v, d, rfl, rfl, rfl, (check_properties hc).1,
               (check_properties hc).2, check_sound hc⟩

end QECLang

namespace QECLang

def RawInSpan (n : Nat) (rows : List (List Bool)) (value : List Bool) : Prop :=
  ∃ (rs : List (Bits n)) (v : Bits n),
    parseRows n rows = some rs ∧ parse n value = some v ∧ InSpan rs v

theorem checkRaw_not_in_span (n : Nat) (rows : List (List Bool)) (value dual : List Bool)
    (h : checkRaw n rows value dual = true) : ¬ RawInSpan n rows value := by
  obtain ⟨rs, v, _, hr, hv, _, _, _, hn⟩ := checkRaw_sound n rows value dual h
  intro ⟨rs', v', hr', hv', hs⟩
  have er : rs = rs' := Option.some.inj (hr.symm.trans hr')
  have ev : v = v' := Option.some.inj (hv.symm.trans hv')
  subst rs'
  subst v'
  exact hn hs

theorem checkRaw_lengths (n : Nat) (rows : List (List Bool)) (value dual : List Bool)
    (h : checkRaw n rows value dual = true) :
    (∀ row ∈ rows, row.length = n) ∧ value.length = n ∧ dual.length = n := by
  obtain ⟨_, _, _, hr, hv, hd, _⟩ := checkRaw_sound n rows value dual h
  exact ⟨parseRows_lengths hr, parse_length hv, parse_length hd⟩

end QECLang
