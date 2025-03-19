/// LRU Cache
#[allow(dead_code)]
struct LRUCache {
    inner: Vec<((i32, i32), i32)>,
    max_size: usize,
}

#[allow(dead_code)]
impl LRUCache {
    fn new(capacity: i32) -> Self {
        assert!(capacity > 0);
        LRUCache {
            inner: Vec::new(),
            max_size: capacity as usize,
        }
    }

    fn get(&mut self, key: i32) -> i32 {
        self.all_add();
        for ((k, v), time) in &mut self.inner {
            if *k == key {
                *time = 0;
                return *v;
            }
        }
        -1
    }

    fn put(&mut self, key: i32, value: i32) {
        self.all_add();
        for ((k, v), time) in &mut self.inner {
            if *k == key {
                *v = value;
                *time = 0;
                return;
            }
        }
        if self.inner.len() >= self.max_size {
            let index = self.find_retire();
            self.inner[index] = ((key, value), 0);
        } else {
            self.inner.push(((key, value), 0));
        }
    }

    fn find_retire(&self) -> usize {
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
    fn all_add(&mut self) {
        for (_, time) in &mut self.inner {
            *time += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lru() {
        let mut cache = LRUCache::new(2);
        cache.put(1, 1);
        cache.put(2, 2);
        assert_eq!(cache.get(1), 1);
        cache.put(3, 3);
        assert_eq!(cache.get(2), -1);
        cache.put(4, 4);
        assert_eq!(cache.get(1), -1);
        assert_eq!(cache.get(3), 3);
        assert_eq!(cache.get(4), 4);
    }
}
