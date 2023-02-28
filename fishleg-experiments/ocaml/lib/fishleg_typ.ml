open Base
include Typ
module Network = Network
module Likelihood = Likelihood
module Approx = Approx

(** main Fishleg parameters *)
type config =
  { learning_rate : int -> float option
  ; learning_rate_aux : int -> float Option.t
  ; bs : int
  ; bs_curvature : int
  ; beta : float
  ; damping : float Option.t
  ; weight_decay : float option
  ; n_aux_iter : int -> int
  ; adam : Prms.Opt.Adam.config
  }

type ('prms, 'g) monitor_fun =
  k:int -> loss:float -> aux_loss:float -> g:'g -> 'prms -> unit

module P = struct
  type ('a, 'b) p =
    { net : 'a
    ; lik : 'b
    }
  [@@deriving prms]
end

(** Type of model that can be optimized using the natural gradient *)
module type T = sig
  module Net : Network.T
  module F : Approx.T with module W = Net.W
  module Lik : Likelihood.T with type stats = Net.output
  module W : Prms.T with type 'a p = ('a Net.W.p, 'a Lik.W.p) P.p
  module L : Prms.T with type 'a p = ('a F.L.p, 'a Lik.L.p) P.p

  type input = Net.input
  type output = Lik.data

  val sample : w:W.t' -> input -> unit -> input * output

  (** main loss *)
  val neg_logp : data:input * output -> W.t' -> AD.t

  (** cross-entropy, locally convex in delta *)
  val cross_entropy : w:W.t' -> input:input -> W.t' -> AD.t

  (** gradient of a quadratic approximation of the Legendre transform *)
  val q_v_prod : lambda:L.t' -> W.t' -> W.t'

  val aux_loop
    :  config:config
    -> (int -> input * output)
    -> k:int
    -> w:W.t
    -> g:(unit -> W.t')
    -> L.t * L.t' Prms.Opt.Adam.state option
    -> float list * (L.t * L.t' Prms.Opt.Adam.state option)

  val value_and_grad_with_info : data:input * output -> w:W.t -> AD.t * W.t'

  (** main FishLeg training loop *)
  val train
    :  ?monitor:(W.t * L.t, W.t') monitor_fun
    -> config:config
    -> w:W.t
    -> lambda:L.t
    -> n_iter:int
    -> (int -> input * output)
    -> W.t

  (** SGD provided for comparison *)
  val train_sgd
    :  ?monitor:(W.t, W.t') monitor_fun
    -> config:Prms.Opt.SGD.config
    -> learning_rate:(int -> float option)
    -> bs:int
    -> w:W.t
    -> n_iter:int
    -> (int -> input * output)
    -> W.t

  (** SGD with momentum provided for comparison *)
  val train_sgdm
    :  ?monitor:(W.t, W.t') monitor_fun
    -> config:Prms.Opt.SGD_momentum.config
    -> learning_rate:(int -> float option)
    -> bs:int
    -> w:W.t
    -> n_iter:int
    -> (int -> input * output)
    -> W.t

  (** Adam provided for comparison *)
  val train_adam
    :  ?monitor:(W.t, W.t') monitor_fun
    -> config:Prms.Opt.Adam.config
    -> learning_rate:(int -> float option)
    -> bs:int
    -> w:W.t
    -> n_iter:int
    -> (int -> input * output)
    -> W.t
end
