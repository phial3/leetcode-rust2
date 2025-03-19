use crate::Solution;

impl Solution {
    pub fn is_array_special(nums: Vec<i32>) -> bool {
        nums.iter()
            .enumerate()
            .all(|(i, &num)| (num % 2) != (i as i32 % 2))
    }
}

#[cfg(test)]
mod tests {
    use crate::Solution;

    #[test]
    fn p3151_test() {
        check(&[], true);
        check(&[1], true);
        check(&[5], true);
        check(&[-3, 4], true);
        check(&[1, 3], false);
        check(&[2, 4], false);
        check(&[2, 1, 4], false);
        check(&[4, 3, 1, 6], false);
    }

    fn check(nums: &[i32], expect: bool) {
        assert_eq!(Solution::is_array_special(nums.to_vec()), expect);
    }
}
