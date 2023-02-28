include module type of Fishleg_typ

(** Model construction by combining the pieces together *)
module Model
    (Net : Network.T)
    (F : Approx.T with type 'a W.p = 'a Net.W.p)
    (Lik : Likelihood.T with type stats = Net.output) :
  T with module Net = Net and module F = F and module Lik = Lik
