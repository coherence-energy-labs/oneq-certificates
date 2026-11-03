import OneqCert.Checker
import OneqCert.Statements

/-! Prints the axioms every certified theorem rests on. A `sorryAx` here is a failure. -/

#print axioms OneqCert.facet_violation
#print axioms OneqCert.facet_inequality
#print axioms OneqCert.sum_coef
#print axioms OneqCert.facet_sum_le
#print axioms OneqCert.safe_bound_le_weight
#print axioms OneqCert.xor_parity
#print axioms OneqCert.xor_list_parity
#print axioms OneqCert.lattice_step
#print axioms OneqCert.sum_on_lattice
#print axioms OneqCert.Cert.accept_sound
#print axioms OneqCert.Statements.accept_sound_pinned
