"""Statistical region validation and global inner-region certification.

Sampling estimates loss and error rates (:mod:`~OpenRC.validation.quality`);
:func:`verify_inner_region` checks that a region is inner either with a
certificate (:func:`certify_inner_region`: global limit bounds, plus existence
proven at every point by :func:`prove_existence` where the contraction
condition holds) or by Monte Carlo.  Both modes decide sampled points with the
escalating :class:`PointVerifier`.
"""

from OpenRC.validation.adjudicate import Adjudication, Adjudicator
from OpenRC.validation.certify import (
    InnerCertificate,
    LimitCheck,
    certify_inner_region,
    shrink_to_certified,
)
from OpenRC.validation.existence import ExistenceProof, prove_existence
from OpenRC.validation.pointcheck import (
    PointOutcome,
    PointVerification,
    PointVerifier,
    WorstViolation,
)
from OpenRC.validation.quality import (
    RegionQuality,
    diagnose_violations,
    evaluate_region,
    wilson_interval,
)
from OpenRC.validation.verify import (
    VERIFY_MODES,
    VerificationReport,
    monte_carlo_cube,
    verify_inner_region,
)
from OpenRC.validation.samplers import (
    SAMPLERS,
    BoxSampler,
    InRegionSampler,
    Sampler,
    ScenarioSampler,
    SobolBoxSampler,
    SurfaceSampler,
    get_sampler,
    list_samplers,
)

__all__ = [
    "Adjudication",
    "Adjudicator",
    "ExistenceProof",
    "InnerCertificate",
    "LimitCheck",
    "PointOutcome",
    "PointVerification",
    "PointVerifier",
    "BoxSampler",
    "InRegionSampler",
    "RegionQuality",
    "SAMPLERS",
    "Sampler",
    "ScenarioSampler",
    "SobolBoxSampler",
    "SurfaceSampler",
    "certify_inner_region",
    "prove_existence",
    "shrink_to_certified",
    "VERIFY_MODES",
    "VerificationReport",
    "WorstViolation",
    "monte_carlo_cube",
    "verify_inner_region",
    "diagnose_violations",
    "evaluate_region",
    "get_sampler",
    "list_samplers",
    "wilson_interval",
]
