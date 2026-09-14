// L2' certified logical-class attestation -- the inline detector, in RTL.
//
// WHAT IT CHECKS, and why it is one cycle of combinational logic:
//
//     violation = pred_A ^ pred_B ^ (^(syndrome & F_MASK))
//
// Two homologically equivalent logical-operator representatives must
// agree, up to the parity of a FIXED subset of detectors -- the
// stabilizers lying between them. F_MASK is that subset, solved offline
// from the DEM and constant at synthesis time, so the runtime cost is an
// AND-reduce and an XOR tree, nothing more.
//
// This is the layer C2's three-layer detector is missing. L1 (syndrome
// consistency) and L2 (LP-dual weight optimality) are both functions of
// the CORRECTION, and an observable-mask upset does not change the
// correction -- so neither can fire, by construction, on the fault
// measured to multiply the logical error rate 17.5x at d=5 and to leave
// the corrupted decoder's error rate essentially independent of distance.
//
// COST: the XOR tree is over POPCOUNT(F_MASK) detector bits, which is
// (d+1)^2/2 -- 8/18/32/50 at d=3/5/7/9. Depth ceil(log2(popcount)) + 2.
// No memory, no state, no multiplier. The area claim in C2 (~5,400 LUTs,
// 0.26% of IBM's 2,106,738-LUT gross-code decoder) is a SYNTHESIS claim
// and is NOT made here: yosys is not installed on this machine, so what
// this file supports is the FUNCTION, verified bit-exact against the
// Python implementation, plus an analytic gate count. Anyone with a
// toolchain can close the remaining gap; nobody should quote a LUT number
// that no synthesizer produced.

`default_nettype none

module l2prime_detector #(
    parameter integer N_DET = 32,          // detectors in the window
    parameter [N_DET-1:0] F_MASK = {N_DET{1'b0}}   // the solved functional
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 valid_in,
    input  wire [N_DET-1:0]     syndrome,  // detector bits this shot
    input  wire                 pred_a,    // class from representative A
    input  wire                 pred_b,    // class from representative B
    output reg                  valid_out,
    output reg                  violation  // 1 => the two derivations
);                                         //      disagree: mask fault

    // The stabilizer parity between the two representatives. A single
    // upset in the per-edge observable table moves pred_a or pred_b but
    // cannot move F_MASK, which is why the check has any power at all.
    wire f_parity = ^(syndrome & F_MASK);
    wire viol_c   = pred_a ^ pred_b ^ f_parity;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 1'b0;
            violation <= 1'b0;
        end else begin
            valid_out <= valid_in;
            violation <= valid_in ? viol_c : 1'b0;
        end
    end

endmodule

`default_nettype wire
