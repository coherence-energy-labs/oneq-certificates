// Testbench: drive the RTL from vectors the PYTHON implementation
// produced, and require bit-exact agreement on every one.
//
// The vectors carry the expected `violation` computed by the Python
// detector, so this is a cross-LANGUAGE differential check, not a
// restatement of the RTL in a testbench. A testbench that recomputed the
// expected value with the same expression the DUT uses would agree with
// itself and prove nothing.

`timescale 1ns/1ps
`default_nettype none

module tb_l2prime;
    localparam integer N_DET = 32;
    localparam integer MAXV  = 20000;

    reg                  clk = 1'b0;
    reg                  rst_n = 1'b0;
    reg                  valid_in = 1'b0;
    reg  [N_DET-1:0]     syndrome = {N_DET{1'b0}};
    reg                  pred_a = 1'b0, pred_b = 1'b0;
    wire                 valid_out, violation;

    reg [N_DET-1:0]      v_syn   [0:MAXV-1];
    reg                  v_a     [0:MAXV-1];
    reg                  v_b     [0:MAXV-1];
    reg                  v_exp   [0:MAXV-1];
    reg [N_DET-1:0]      fmask;
    integer              nvec, i, errors, fired;

    // F_MASK arrives as a parameter, so it is read from the vector file
    // header and passed in by regenerating this bench per case. Here the
    // mask is applied in the DUT instantiation below.
    l2prime_detector #(.N_DET(N_DET), .F_MASK(32'hDEADBEEF)) dut (
        .clk(clk), .rst_n(rst_n), .valid_in(valid_in),
        .syndrome(syndrome), .pred_a(pred_a), .pred_b(pred_b),
        .valid_out(valid_out), .violation(violation)
    );

    always #5 clk = ~clk;

    initial begin
        errors = 0;
        fired  = 0;
        $readmemh("vec_syn.hex", v_syn);
        $readmemb("vec_a.bin",   v_a);
        $readmemb("vec_b.bin",   v_b);
        $readmemb("vec_exp.bin", v_exp);
        nvec = 0;
        for (i = 0; i < MAXV; i = i + 1)
            if (v_exp[i] !== 1'bx) nvec = i + 1;

        @(negedge clk); rst_n = 1'b1;

        for (i = 0; i < nvec; i = i + 1) begin
            @(negedge clk);
            syndrome = v_syn[i];
            pred_a   = v_a[i];
            pred_b   = v_b[i];
            valid_in = 1'b1;
            @(posedge clk);
            @(negedge clk);
            valid_in = 1'b0;
            if (valid_out !== 1'b1) begin
                $display("VEC %0d: valid_out did not assert", i);
                errors = errors + 1;
            end else if (violation !== v_exp[i]) begin
                $display("VEC %0d: RTL %b != PYTHON %b (syn=%h a=%b b=%b)",
                         i, violation, v_exp[i], v_syn[i], v_a[i], v_b[i]);
                errors = errors + 1;
            end
            if (v_exp[i] === 1'b1) fired = fired + 1;
        end

        $display("VECTORS %0d  FIRED %0d  ERRORS %0d", nvec, fired, errors);
        // A run in which the detector never fires agrees trivially. The
        // vectors are generated to include both outcomes; if they did
        // not, this bench would pass while testing one branch.
        if (fired == 0)
            $display("VACUOUS: no vector expected a violation");
        else if (fired == nvec)
            $display("VACUOUS: every vector expected a violation");
        $finish;
    end
endmodule

`default_nettype wire
