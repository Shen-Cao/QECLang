import Std

namespace QECLang

/-- A finite binary vector whose width is part of its type. -/
inductive Bits : Nat → Type where
  | nil : Bits 0
  | cons : Bool → Bits n → Bits (Nat.succ n)
  deriving Repr, DecidableEq

def zero : (n : Nat) → Bits n
  | 0 => .nil
  | n + 1 => .cons false (zero n)

def xor : Bits n → Bits n → Bits n
  | .nil, .nil => .nil
  | .cons a as, .cons b bs => .cons (a ^^ b) (xor as bs)

def dot : Bits n → Bits n → Bool
  | .nil, .nil => false
  | .cons a as, .cons b bs => (a && b) ^^ dot as bs

theorem dot_zero (d : Bits n) : dot (zero n) d = false := by
  induction d with
  | nil => rfl
  | cons b ds ih =>
    change (false ^^ dot (zero _) ds) = false
    rw [ih]
    rfl

theorem dot_xor (a b d : Bits n) : dot (xor a b) d = (dot a d ^^ dot b d) := by
  induction a with
  | nil => cases b; cases d; rfl
  | cons x xs ih =>
    cases b with
    | cons y ys =>
      cases d with
      | cons z zs =>
        change (((x ^^ y) && z) ^^ dot (xor xs ys) zs) =
          (((x && z) ^^ dot xs zs) ^^ ((y && z) ^^ dot ys zs))
        rw [ih]
        cases x <;> cases y <;> cases z <;>
          cases dot xs zs <;> cases dot ys zs <;> rfl

/-- Membership in the XOR span: start at zero and add generator rows. -/
inductive InSpan (rows : List (Bits n)) : Bits n → Prop where
  | zero : InSpan rows (zero n)
  | add {row value} : row ∈ rows → InSpan rows value → InSpan rows (xor row value)

def Orthogonal (rows : List (Bits n)) (dual : Bits n) : Prop :=
  ∀ row, row ∈ rows → dot row dual = false

theorem span_pairs_zero {rows : List (Bits n)} {dual value : Bits n}
    (hrows : Orthogonal rows dual) (hspan : InSpan rows value) :
    dot value dual = false := by
  induction hspan with
  | zero => exact dot_zero dual
  | add hrow _ ih => rw [dot_xor, hrows _ hrow, ih]; rfl

theorem odd_pairing_not_in_span {rows : List (Bits n)} {dual value : Bits n}
    (hrows : Orthogonal rows dual) (hodd : dot value dual = true) :
    ¬ InSpan rows value := by
  intro hspan
  have heven := span_pairs_zero hrows hspan
  rw [heven] at hodd
  contradiction

end QECLang
