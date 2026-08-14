use crate::discrete_factorizations::{
    BooleanFactorization, BooleanFunction, BooleanFunctionOps, Factorizer,
};

/// An integer distribution represented as a vector of BDD bits.
/// Each bit is a BDD representing the probability distribution over that bit position.
#[derive(Debug, Clone)]
pub struct IntDist {
    pub bits: Vec<BooleanFunction>,
}

impl IntDist {
    pub fn new(bits: Vec<BooleanFunction>) -> Self {
        IntDist { bits }
    }

    pub fn width(&self) -> usize {
        self.bits.len()
    }

    /// Get the BDD for bit i, returning false_ptr() (zero) for out-of-range bits.
    pub fn bit(&self, i: usize) -> BooleanFunction {
        if i < self.bits.len() {
            self.bits[i].clone()
        } else {
            BooleanFunction::false_ptr()
        }
    }
}

/// Check equality of two IntDists: AND over (x_i IFF y_i) for each bit.
pub fn int_dist_eq(x: &IntDist, y: &IntDist, builder: &Factorizer) -> BooleanFunction {
    assert_eq!(x.width(), y.width());

    let mut result = BooleanFunction::true_ptr();
    for i in 0..x.width() {
        let iff = builder.iff(&x.bits[i], &y.bits[i]);
        result = builder.and(&result, &iff);
        if result.is_false() {
            return BooleanFunction::false_ptr();
        }
    }

    result
}

/// Get the BDD for a specific integer value within an IntDist.
/// The value is encoded in binary (LSB first).
pub fn int_dist_at_int(val: &IntDist, i: u64, builder: &Factorizer) -> BooleanFunction {
    let mut bf = BooleanFunction::true_ptr();
    for bit_idx in 0..val.width() {
        let bit_val = (i >> bit_idx) & 1 == 1;
        let bit_formula = &val.bits[bit_idx];
        if bit_val {
            bf = builder.and(&bf, bit_formula);
        } else {
            let neg = builder.negate(bit_formula);
            bf = builder.and(&bf, &neg);
        }
    }
    bf
}

/// Enumerate all 2^n possible values of an IntDist.
/// Returns a list of (integer_value, bf) pairs.
pub fn enumerate_int_dist(
    val: &IntDist,
    guard: &BooleanFunction,
    builder: &Factorizer,
) -> Vec<(u64, BooleanFunction)> {
    let n = val.width();
    let mut worlds = Vec::new();
    for i in 0..(1u64 << n) {
        let world_bf = builder.and(&int_dist_at_int(val, i, builder), guard);
        if !world_bf.is_false() {
            worlds.push((i, world_bf));
        }
    }
    worlds
}

/// Wrapping addition of two IntDists (mod 2^n where n = max width).
/// Implements a ripple-carry adder; the final carry is discarded.
pub fn int_dist_add(a: &IntDist, b: &IntDist, builder: &Factorizer) -> IntDist {
    assert_eq!(a.width(), b.width());
    let n = a.width();

    let mut result_bits = Vec::with_capacity(n);
    let mut carry = BooleanFunction::false_ptr();

    for i in 0..n {
        let ai = a.bit(i);
        let bi = b.bit(i);

        // sum_i = a_i XOR b_i XOR carry
        let a_xor_b = builder.xor(&ai, &bi);
        let sum_i = builder.xor(&a_xor_b, &carry);
        result_bits.push(sum_i);

        // carry = ITE(a_i, b_i OR carry, b_i AND carry)
        let b_or_carry = builder.or(&bi, &carry);
        let b_and_carry = builder.and(&bi, &carry);
        carry = builder.ite(&ai, &b_or_carry, &b_and_carry);
    }

    IntDist::new(result_bits)
}

/// Wrapping subtraction of two IntDists: a - b (mod 2^n where n = max width).
/// Implements a ripple-borrow subtractor; the final borrow is discarded.
pub fn int_dist_sub(a: &IntDist, b: &IntDist, builder: &Factorizer) -> IntDist {
    assert_eq!(a.width(), b.width());
    let n = a.width();

    let mut result_bits = Vec::with_capacity(n);
    let mut borrow = BooleanFunction::false_ptr();

    for i in 0..n {
        let ai = a.bit(i);
        let bi = b.bit(i);

        // diff_i = a_i XOR b_i XOR borrow
        let a_xor_b = builder.xor(&ai, &bi);
        let diff_i = builder.xor(&a_xor_b, &borrow);
        result_bits.push(diff_i);

        // borrow = ITE(a_i, b_i AND borrow, b_i OR borrow)
        let b_and_borrow = builder.and(&bi, &borrow);
        let b_or_borrow = builder.or(&bi, &borrow);
        borrow = builder.ite(&ai, &b_and_borrow, &b_or_borrow);
    }

    IntDist::new(result_bits)
}

/// Unsigned less-than comparison: returns a BDD representing a < b.
/// Uses LSB-to-MSB recurrence:
///   lt_i = ITE(a_i, b_i AND lt_{i-1}, b_i OR lt_{i-1})
pub fn int_dist_lt(a: &IntDist, b: &IntDist, builder: &Factorizer) -> BooleanFunction {
    assert_eq!(a.width(), b.width());
    let n = a.width();

    let mut lt = BooleanFunction::false_ptr();

    for i in 0..n {
        let ai = a.bit(i);
        let bi = b.bit(i);

        let b_and_lt = builder.and(&bi, &lt);
        let b_or_lt = builder.or(&bi, &lt);
        lt = builder.ite(&ai, &b_and_lt, &b_or_lt);
    }

    lt
}
