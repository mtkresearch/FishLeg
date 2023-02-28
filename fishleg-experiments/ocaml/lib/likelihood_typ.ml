open Base
open Typ

module type T = sig
  module W : Prms.T
  module L : Prms.T

  (** sufficient statistics, will be the output of the upstream network *)
  type stats

  (** type of data / observations we are modelling with the likelihood *)
  type data

  (** samples an output given the sufficient statistics *)
  val sample : w:W.t' -> stats -> data

  (** negative log likelihood associated with a pair of sufficient stats and observation *)
  val neg_logp : w:W.t' -> obs:data -> stats -> AD.t

  (** for some likelihoods it might be possible to analytically calculate the
      expectation of [neg_logp ~w (stats, data)]
      where data is drawn as [sample ~w:w' stats'] *)
  val expected_neg_logp
    : (expectation_under:stats * W.t' -> stats * W.t' -> AD.t) Option.t

  (** for likelihoods that introduce parameters of their own, this function will
      provide their contribution to the overall (quadratic) approximation of the
      Legendre transform; note that his way of structuring code will prevent us
      from leveraging cross-curvature between the parameters of the likelihood and
      those of the main model, but that's probably OK -- it's unclear how to model
      those dependencies generically anyways. Nevertheless, this could be suboptimal
      for training e.g. Gaussian process-based models, where there is typically a
      lot of covariance between the kernel parameters and the noise variance... *)
  val q_v_prod : lambda:AD.t L.p -> W.t' -> W.t'
end
