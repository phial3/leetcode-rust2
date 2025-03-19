use crate::Solution;
use std::{cell::RefCell, rc::Rc, vec};

/// `ListNode` definition in LeetCode
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

/// `TreeNode` definition in LeetCode
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

impl Solution {
    // pub fn middle_node(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
    //     let mut head = head;
    //     let mut fast = &head;
    //     let mut slow = &head;
    //     loop {
    //         // fast pointer move 2 steps
    //         if let Some(node) = &fast {
    //             fast = &node.next;
    //         } else {
    //             break;
    //         }
    //         if let Some(node) = &fast {
    //             fast = &node.next;
    //         } else {
    //             break;
    //         }
    //         // slow pointer move 1 step
    //         if let Some(node) = &slow {
    //             slow = &node.next;
    //         } else {
    //             break;
    //         }
    //     }
    //     let mid_addr = if let Some(node) = slow {
    //         node.as_ref() as *const ListNode
    //     } else {
    //         return None;
    //     };
    //     while let Some(node) = head.as_ref() {
    //         let addr = node.as_ref() as *const ListNode;
    //         if addr != mid_addr {
    //             head = head.unwrap().next;
    //         } else {
    //             break;
    //         }
    //     }
    //     head
    // }

    pub fn middle_node(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut slow = head.clone();
        let mut fast = head.clone();
        while fast.is_some() && fast.as_ref().unwrap().next.is_some() {
            slow = slow.unwrap().next;
            fast = fast.unwrap().next.unwrap().next;
        }
        slow
    }

    pub fn remove_elements(head: Option<Box<ListNode>>, val: i32) -> Option<Box<ListNode>> {
        let mut head = head;
        let mut res_head = Box::new(ListNode::new(0));
        let mut pos = &mut res_head;
        while let Some(mut node) = head.take() {
            head = node.next.take();
            if node.val != val {
                pos = pos.next.get_or_insert(node);
            }
        }
        res_head.next
    }

    pub fn delete_duplicates(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut head = head;
        let mut res_head = Box::new(ListNode::new(0));
        let mut prev: Option<&mut Box<ListNode>> = None;
        let mut curr = &mut res_head;
        while let Some(mut node) = head.take() {
            head = node.next.take();
            if let Some(prev_node) = prev {
                if node.val != prev_node.val {
                    curr = curr.next.get_or_insert(node);
                }
            } else {
                curr = curr.next.get_or_insert(node);
            }
            prev = Some(curr);
        }
        res_head.next
    }

    pub fn delete_duplicates_2(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        if head.is_none() || head.as_ref().unwrap().next.is_none() {
            return head;
        }
        let mut head = head;
        let mut res_head = Box::new(ListNode::new(0));
        let mut prev = None;
        let mut curr = &mut res_head;
        let mut num = 0;
        while let Some(mut node) = head.take() {
            head = node.next.take();
            if prev.is_none() {
                prev = Some(node);
            } else if node.val != prev.as_ref().unwrap().val {
                if num == 0 {
                    let prev_node = prev.take().unwrap();
                    curr = curr.next.get_or_insert(prev_node);
                }
                if head.is_none() {
                    curr = curr.next.get_or_insert(node);
                } else {
                    prev = Some(node);
                    num = 0;
                }
            } else {
                num += 1;
            }
        }
        res_head.next
    }

    pub fn swap_pairs(head: Option<Box<ListNode>>) -> Option<Box<ListNode>> {
        let mut head = head;
        let mut res_head = Box::new(ListNode::new(0));
        let mut pos = &mut res_head;
        while head.is_some() && head.as_mut().unwrap().next.is_some() {
            let mut prev = head.take().unwrap();
            head = prev.next.take();
            let mut after = head.take().unwrap();
            head = after.next.take();
            pos = pos.next.get_or_insert(after);
            pos = pos.next.get_or_insert(prev);
        }
        if let Some(node) = head.take() {
            pos.next.get_or_insert(node);
        }
        res_head.next
    }

    pub fn rotate_right(head: Option<Box<ListNode>>, k: i32) -> Option<Box<ListNode>> {
        if head.is_none() {
            return head;
        }
        let mut head = head;
        let mut head_pos = head.as_ref();
        let mut res_head = Box::new(ListNode::new(0));
        let mut pos = &mut res_head;
        let mut len = 0;
        while let Some(node) = head_pos {
            head_pos = node.next.as_ref();
            len += 1;
        }
        let i = k % len;
        for _ in 0..(len - i) {
            if let Some(mut node) = head.take() {
                head = node.next.take();
                pos = pos.next.get_or_insert(node);
            } else {
                panic!()
            }
        }
        pos = &mut res_head;
        while let Some(mut node) = head.take() {
            head = node.next.take();
            let temp = pos.next.take();
            pos = pos.next.get_or_insert(node);
            if let Some(node) = temp {
                pos.next.get_or_insert(node);
            }
        }
        res_head.next
    }

    pub fn inver(root: &mut Option<Rc<RefCell<TreeNode>>>) {
        if let Some(node) = root {
            let mut node = node.borrow_mut();
            Self::inver(&mut node.left);
            Self::inver(&mut node.right);

            let left = node.left.take();
            let right = std::mem::replace(&mut node.right, left);
            let _ = std::mem::replace(&mut node.left, right);
        }
    }

    pub fn is_symmetric(root: Option<Rc<RefCell<TreeNode>>>) -> bool {
        fn recursive(p: Option<&Rc<RefCell<TreeNode>>>, q: Option<&Rc<RefCell<TreeNode>>>) -> bool {
            match (p, q) {
                (None, None) => true,
                (Some(left), Some(right)) => {
                    let left = left.borrow();
                    let right = right.borrow();
                    left.val == right.val
                        && recursive(left.left.as_ref(), right.right.as_ref())
                        && recursive(left.right.as_ref(), right.left.as_ref())
                }
                _ => false,
            }
        }
        match root {
            None => true,
            Some(node) => {
                let node = node.borrow();
                recursive(node.left.as_ref(), node.right.as_ref())
            }
        }
    }

    pub fn merge_trees(
        mut root1: Option<Rc<RefCell<TreeNode>>>,
        mut root2: Option<Rc<RefCell<TreeNode>>>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        fn recursive(p: &mut Option<Rc<RefCell<TreeNode>>>, q: &mut Option<Rc<RefCell<TreeNode>>>) {
            match (&p, &q) {
                (Some(left), Some(right)) => {
                    let mut left = left.borrow_mut();
                    let mut right = right.borrow_mut();
                    left.val += right.val;
                    recursive(&mut left.left, &mut right.left);
                    recursive(&mut left.right, &mut right.right);
                }
                (None, Some(_)) => {
                    *p = q.take();
                }
                _ => {}
            }
        }
        recursive(&mut root1, &mut root2);
        root1
    }

    pub fn search_bst(
        root: Option<Rc<RefCell<TreeNode>>>,
        val: i32,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        fn recursive(
            node: &Option<Rc<RefCell<TreeNode>>>,
            val: i32,
        ) -> Option<Rc<RefCell<TreeNode>>> {
            if let Some(n) = &node {
                if n.borrow().val == val {
                    return node.clone();
                }
                if n.borrow().val < val {
                    return recursive(&n.borrow().right, val);
                } else {
                    return recursive(&n.borrow().left, val);
                }
            } else {
                None
            }
        }
        recursive(&root, val)
    }

    pub fn prune_tree(root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        fn need_remove(root: Option<&Rc<RefCell<TreeNode>>>) -> bool {
            match root {
                None => true,
                Some(node) => {
                    if node.borrow().val == 1 {
                        false
                    } else {
                        need_remove(node.borrow().left.as_ref())
                            && need_remove(node.borrow().right.as_ref())
                    }
                }
            }
        }
        fn new_tree(root: Option<&Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
            if !need_remove(root) {
                if let Some(node) = root {
                    let new_root = Rc::new(RefCell::new(TreeNode::new(node.borrow().val)));
                    new_root.borrow_mut().left = new_tree(node.borrow().left.as_ref());
                    new_root.borrow_mut().right = new_tree(node.borrow().right.as_ref());
                    Some(new_root)
                } else {
                    None
                }
            } else {
                None
            }
        }
        new_tree(root.as_ref())
    }

    pub fn convert_bst(mut root: Option<Rc<RefCell<TreeNode>>>) -> Option<Rc<RefCell<TreeNode>>> {
        fn recursive(root: &mut Option<Rc<RefCell<TreeNode>>>, sum: &mut i32) {
            if let Some(node) = root {
                let mut node = node.borrow_mut();
                recursive(&mut node.right, sum);
                node.val += *sum;
                *sum = node.val;
                recursive(&mut node.left, sum);
            }
        }
        let mut sum = 0;
        recursive(&mut root, &mut sum);
        root
    }

    pub fn construct_maximum_binary_tree(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
        fn recursive(nums: Vec<i32>) -> Option<Rc<RefCell<TreeNode>>> {
            if nums.is_empty() {
                return None;
            }
            let mut max = nums[0];
            let mut max_pos = 0;
            for (i, _item) in nums.iter().enumerate() {
                if nums[i] > max {
                    max = nums[i];
                    max_pos = i;
                }
            }

            let root = Rc::new(RefCell::new(TreeNode::new(max)));
            root.borrow_mut().left = recursive(nums[0..max_pos].to_vec());
            if max_pos < nums.len() - 1 {
                root.borrow_mut().right = recursive(nums[(max_pos + 1)..].to_vec());
            }
            Some(root)
        }

        recursive(nums)
    }

    pub fn int_to_roman(num: i32) -> String {
        // ref: https://leetcode.com/problems/integer-to-roman/discuss/1016135/Rust%3A-vector-solution
        let m = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1];
        let n = [
            "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I",
        ];

        let (mut num, mut s) = (num, String::new());
        for i in 0..13 {
            let mut j = num / m[i];
            num %= m[i];
            while j > 0 {
                s.push_str(n[i]);
                j -= 1;
            }
        }
        s
    }

    #[allow(non_snake_case)]
    pub fn LRU(&self, operators: Vec<Vec<i32>>, k: i32) -> Vec<i32> {
        assert!(k > 0);
        struct Lru {
            inner: Vec<((i32, i32), i32)>,
            max_size: usize,
        }
        impl Lru {
            pub fn new(max_size: usize) -> Self {
                Self {
                    inner: Vec::new(),
                    max_size,
                }
            }
            pub fn find_retire(&self) -> usize {
                let mut index = 0;
                let mut longest = self.inner[0].1;
                for (i, (_, time)) in self.inner.iter().enumerate() {
                    if *time > longest {
                        longest = *time;
                        index = i;
                    }
                }
                index
            }
            pub fn all_add(&mut self) {
                for (_, time) in &mut self.inner {
                    *time += 1;
                }
            }
            pub fn set(&mut self, val: (i32, i32)) {
                self.all_add();
                if self.inner.len() >= self.max_size {
                    let index = self.find_retire();
                    self.inner[index] = (val, 0);
                } else {
                    self.inner.push((val, 0));
                }
            }
            pub fn get(&mut self, key: i32) -> i32 {
                self.all_add();
                for ((k, v), time) in &mut self.inner {
                    if *k == key {
                        *time = 0;
                        return *v;
                    }
                }
                -1
            }
        }
        let mut res = Vec::new();
        let mut lru = Lru::new(k as usize);
        for ops in operators {
            assert!(ops.len() >= 2);
            match ops[0] {
                1 => {
                    assert_eq!(ops.len(), 3);
                    lru.set((ops[1], ops[2]));
                }
                2 => {
                    assert_eq!(ops.len(), 2);
                    res.push(lru.get(ops[1]));
                }
                _ => panic!("error input!"),
            }
        }
        res
    }

    pub fn convert(s: String, num_rows: i32) -> String {
        if num_rows == 1 {
            return s;
        }
        let mut res_s = String::new();
        let mut res = Vec::new();
        for _ in 0..num_rows {
            res.push(Vec::new());
        }
        let mut pos = 0;
        let mut up = false;
        let mut data = s.as_bytes().to_vec();
        while !data.is_empty() {
            let c = data.remove(0);
            res[pos].push(c);
            if pos == 0 {
                pos += 1;
                up = false;
            } else if pos == num_rows as usize - 1 {
                pos -= 1;
                up = true;
            } else if up {
                pos -= 1;
            } else {
                pos += 1;
            }
        }
        for v in res {
            for c in v {
                res_s.push(c as char);
            }
        }
        res_s
    }

    pub fn letter_combinations(digits: String) -> Vec<String> {
        use std::collections::HashMap;
        fn backtrack(
            res_vec: &mut Vec<String>,
            map: &HashMap<char, String>,
            digits: &String,
            index: usize,
            res: &mut String,
        ) {
            if index == digits.len() {
                res_vec.push(String::from(res.as_str()));
            } else {
                let digit = digits.as_bytes()[index] as char;
                if let Some(s) = map.get(&digit) {
                    let new_s = String::from(s.as_str());
                    for ch in new_s.chars() {
                        res.push(ch);
                        backtrack(res_vec, map, digits, index + 1, res);
                        res.remove(index);
                    }
                }
            }
        }
        if digits.is_empty() {
            return Vec::new();
        }
        let mut map: HashMap<char, String> = HashMap::new();
        map.insert('2', "abc".to_string());
        map.insert('3', "def".to_string());
        map.insert('4', "ghi".to_string());
        map.insert('5', "jkl".to_string());
        map.insert('6', "mno".to_string());
        map.insert('7', "pqrs".to_string());
        map.insert('8', "tuv".to_string());
        map.insert('9', "wxyz".to_string());
        let mut res_vec = Vec::new();
        let mut res = String::new();
        backtrack(&mut res_vec, &map, &digits, 0, &mut res);
        res_vec
    }

    pub fn add_two_numbers(
        l1: Option<Box<ListNode>>,
        l2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        let mut head_1 = l1;
        let mut head_2 = l2;
        let mut res = ListNode::new(0);
        let mut pos = &mut res;
        let mut cin = 0;
        // while let (Some(mut node_1), Some(mut node_2)) = (head_1.take(), head_2.take()) {
        //     head_1 = node_1.next.take();
        //     head_2 = node_2.next.take();
        //     let mut sum = cin + node_1.val + node_2.val;
        //     cin = sum / 10;
        //     sum %= 10;
        //     let node = ListNode::new(sum);
        //     pos = pos.next.get_or_insert(Box::new(node));
        // }
        // while let Some(mut node) = head_1.take() {
        //     head_1 = node.next.take();
        //     let mut sum = cin + node.val;
        //     cin = sum / 10;
        //     sum %= 10;
        //     let new_node = ListNode::new(sum);
        //     pos = pos.next.get_or_insert(Box::new(new_node));
        // }
        // while let Some(mut node) = head_2.take() {
        //     head_2 = node.next.take();
        //     let mut sum = cin + node.val;
        //     cin = sum / 10;
        //     sum %= 10;
        //     let new_node = ListNode::new(sum);
        //     pos = pos.next.get_or_insert(Box::new(new_node));
        // }
        while head_1.is_some() || head_2.is_some() {
            let (mut val_1, mut val_2) = (0, 0);
            if let Some(mut node_1) = head_1.take() {
                head_1 = node_1.next.take();
                val_1 = node_1.val;
            }
            if let Some(mut node_2) = head_2.take() {
                head_2 = node_2.next.take();
                val_2 = node_2.val;
            }
            let mut sum = cin + val_1 + val_2;
            cin = sum / 10;
            sum %= 10;
            let node = ListNode::new(sum);
            pos = pos.next.get_or_insert(Box::new(node));
        }
        if cin == 1 {
            let node = ListNode::new(cin);
            pos.next.get_or_insert(Box::new(node));
        }
        res.next
    }

    pub fn get_winner(arr: Vec<i32>, k: i32) -> i32 {
        if k == 1 {
            return arr[0].max(arr[1]);
        }
        let mut count = 1;
        let mut prev;
        let mut curr;
        if arr[0] > arr[1] {
            prev = 0;
            curr = 2;
        } else {
            prev = 1;
            curr = 2;
        }
        while curr < arr.len() {
            if arr[curr] > arr[prev] {
                prev = curr;
                count = 1;
            } else {
                count += 1;
            }
            if count >= k {
                return arr[prev];
            }
            curr += 1;
        }
        arr[prev]
    }

    pub fn three_sum_closest(mut nums: Vec<i32>, target: i32) -> i32 {
        let len = nums.len();
        nums.sort();
        assert!(len > 2);
        let mut res = nums[0] + nums[1] + nums[2];
        for i in 0..len - 2 {
            let mut l = i + 1;
            let mut r = len - 1;
            while l < r {
                let temp_res = nums[i] + nums[l] + nums[r];
                if temp_res == target {
                    return temp_res;
                } else if temp_res < target {
                    if target - temp_res < (res - target).abs() {
                        res = temp_res;
                    }
                    l += 1;
                } else {
                    if temp_res - target < (res - target).abs() {
                        res = temp_res;
                    }
                    r -= 1;
                }
            }
        }
        res
    }

    pub fn generate_parenthesis(n: i32) -> Vec<String> {
        fn generate(n: i32) -> Vec<String> {
            let mut idx = 0;
            let mut v = Vec::new();
            while idx < n * 2 {
                if idx == 0 {
                    v.push(String::from("("));
                    v.push(String::from(")"));
                } else {
                    let mut temp_v: Vec<String> = v
                        .iter()
                        .map(|s| {
                            let new_s = format!("{})", s);
                            new_s
                        })
                        .collect();
                    v.iter_mut().for_each(|s| s.push('('));
                    while let Some(s) = temp_v.pop() {
                        v.push(s);
                    }
                }
                idx += 1;
            }
            v
        }
        fn is_ok(s: &String) -> bool {
            let mut stack = Vec::new();
            let mut data = s.as_bytes().to_vec();
            loop {
                if !data.is_empty() {
                    let c = data.remove(0);
                    if c == b'(' {
                        stack.push(c);
                    } else if let Some(d) = stack.pop() {
                        if d == b'(' {
                            // drop 掉这对
                        } else {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else {
                    return stack.is_empty();
                }
            }
        }
        let v = generate(n);
        v.iter()
            .filter(|s| is_ok(s))
            .map(|s| String::from(s.as_str()))
            .collect()
    }

    pub fn inorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        fn recursive(root: &Option<Rc<RefCell<TreeNode>>>, res: &mut Vec<i32>) {
            if let Some(node) = root {
                let left = &node.borrow().left;
                let right = &node.borrow().right;
                recursive(left, res);
                let val = node.borrow().val;
                res.push(val);
                recursive(right, res);
            }
        }
        let mut res = Vec::new();
        recursive(&root, &mut res);
        res
    }

    pub fn postorder_traversal(root: Option<Rc<RefCell<TreeNode>>>) -> Vec<i32> {
        // fn recursive(root: &Option<Rc<RefCell<TreeNode>>>, res: &mut Vec<i32>) {
        //     if let Some(node) = root {
        //         let left = &node.borrow().left;
        //         let right = &node.borrow().right;
        //         recursive(left, res);
        //         recursive(right, res);
        //         let val = node.borrow().val;
        //         res.push(val);
        //     }
        // }

        // let mut res = Vec::new();
        // recursive(&root, &mut res);
        // res

        // Ref: https://leetcode-cn.com/problems/binary-tree-postorder-traversal/solution/rust-17xing-by-qweytr_1/
        if root.is_none() {
            return Vec::new();
        }
        let mut res = Vec::new();
        let mut stack = Vec::new();
        if let Some(root) = root {
            stack.push(root);
        }
        while let Some(root) = stack.pop() {
            let mut node = root.borrow_mut();
            res.push(node.val);
            if let Some(x) = node.left.take() {
                stack.push(x);
            }
            if let Some(x) = node.right.take() {
                stack.push(x);
            }
        }
        res.reverse();
        res
    }

    pub fn my_pow(x: f64, n: i32) -> f64 {
        fn quick_pow(x: f64, n: i32) -> f64 {
            if n == 0 {
                return 1.0;
            }
            let y = quick_pow(x, n / 2);
            if n % 2 == 0 {
                y * y
            } else {
                y * y * x
            }
        }
        if n >= 0 {
            quick_pow(x, n)
        } else {
            1.0 / quick_pow(x, n)
        }
    }

    pub fn construct_arr(a: Vec<i32>) -> Vec<i32> {
        let len = a.len();
        let mut b = vec![1; len];
        let mut t = 1;
        for i in 1..len {
            b[i] = b[i - 1] * a[i - 1];
        }
        for i in 1..len {
            t *= a[len - i];
            b[len - 1 - i] *= t;
        }
        b
    }

    pub fn min_path_sum(grid: Vec<Vec<i32>>) -> i32 {
        let len_i = grid.len();
        if len_i == 0 {
            return 0;
        }
        let len_j = grid[0].len();
        let mut dp = vec![vec![0; len_j]; len_i];
        dp[0][0] = grid[0][0];
        for i in 1..len_i {
            dp[i][0] = dp[i - 1][0] + grid[i][0];
        }
        for j in 1..len_j {
            dp[0][j] = dp[0][j - 1] + grid[0][j];
        }
        for i in 1..len_i {
            for j in 1..len_j {
                dp[i][j] = dp[i - 1][j].min(dp[i][j - 1]) + grid[i][j];
            }
        }
        dp[len_i - 1][len_j - 1]
    }

    pub fn unique_paths_with_obstacles(obstacle_grid: Vec<Vec<i32>>) -> i32 {
        let m = obstacle_grid.len();
        let n = obstacle_grid[0].len();
        if m == 1 {
            if obstacle_grid[0].contains(&1) {
                return 0;
            } else {
                return 1;
            }
        }
        if n == 1 {
            if obstacle_grid.contains(&vec![1]) {
                return 0;
            } else {
                return 1;
            }
        }
        if obstacle_grid[0][0] == 1 {
            return 0;
        }
        // 动态规划
        let mut dp = vec![vec![0; n]; m];
        for i in 0..n {
            if obstacle_grid[0][i] == 1 {
                dp[0][i..].iter_mut().for_each(|x| *x = 0);
                break;
            } else {
                dp[0][i] = 1;
            }
        }
        for i in 0..m {
            if obstacle_grid[i][0] == 1 {
                dp[i..].iter_mut().for_each(|v| v[0] = 0);
                break;
            } else {
                dp[i][0] = 1;
            }
        }
        for i in 1..m {
            for j in 1..n {
                if obstacle_grid[i][j] == 1 {
                    dp[i][j] = 0;
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        dp[m - 1][n - 1]
    }

    pub fn is_interleave(s1: String, s2: String, s3: String) -> bool {
        let n = s1.len();
        let m = s2.len();
        let t = s3.len();
        if m + n != t {
            return false;
        }
        let mut dp = vec![vec![false; m + 1]; n + 1];
        dp[0][0] = true;
        for i in 0..n + 1 {
            for j in 0..m + 1 {
                let p = i + j - 1;
                if i > 0 {
                    dp[i][j] =
                        dp[i][j] || dp[i - 1][j] && (s1.as_bytes()[i - 1] == s3.as_bytes()[p]);
                }
                if j > 0 {
                    dp[i][j] =
                        dp[i][j] || dp[i][j - 1] && (s2.as_bytes()[j - 1] == s3.as_bytes()[p]);
                }
            }
        }
        dp[n][m]
    }

    pub fn minimum_total(triangle: Vec<Vec<i32>>) -> i32 {
        let mut dp = triangle.clone();
        for i in 0..dp.len() {
            if i == 0 {
                continue;
            }
            let t = dp[i].len();
            for j in 0..t {
                if j == 0 {
                    dp[i][j] += dp[i - 1][j];
                } else if j == t - 1 {
                    dp[i][j] += dp[i - 1][j - 1];
                } else {
                    dp[i][j] += dp[i - 1][j - 1].min(dp[i - 1][j]);
                }
            }
        }
        let mut res = dp[dp.len() - 1][0];
        dp[dp.len() - 1].iter().for_each(|i| {
            if *i < res {
                res = *i;
            }
        });
        res
    }

    pub fn remove_duplicates(nums: &mut [i32]) -> i32 {
        if nums.is_empty() || nums.len() == 1 {
            return nums.len() as i32;
        }
        let len = nums.len();
        let mut slow = 2usize;
        let mut fast = 2usize;
        while fast < len {
            if nums[slow - 2] != nums[fast] {
                nums[slow] = nums[fast];
                slow += 1;
            }
            fast += 1;
        }
        slow as i32
    }
}
