//! Disjoint-set (union-find) support for node clusterings.

use indexmap::{IndexMap, IndexSet};
use petgraph::{graph::IndexType, unionfind::UnionFind};

/// A node clustering, mapping each set representative to its set of nodes.
pub type Clustering<K> = IndexMap<K, IndexSet<K>>;

/// A disjoint-set data structure over the elements `0..n`.
#[derive(Debug, Clone)]
pub struct DisjointSets<K> {
    union_find: UnionFind<K>,
    n: usize,
}

impl<K: IndexType> DisjointSets<K> {
    /// Constructs a new `DisjointSets` with `n` singleton sets.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of elements.
    ///
    /// # Returns
    ///
    /// The structure with every element in its own set.
    pub fn new(n: usize) -> Self {
        Self {
            union_find: UnionFind::new(n),
            n,
        }
    }

    /// Returns the representative of the set containing `x`.
    ///
    /// # Arguments
    ///
    /// * `x` - The element to look up.
    ///
    /// # Returns
    ///
    /// The set representative.
    pub fn repr(&self, x: K) -> K {
        self.union_find.find(x)
    }

    /// Merges the sets containing `x` and `y`.
    ///
    /// # Arguments
    ///
    /// * `x` - An element of the first set.
    /// * `y` - An element of the second set.
    ///
    /// # Returns
    ///
    /// `false` if `x` and `y` were already in the same set.
    pub fn union(&mut self, x: K, y: K) -> bool {
        self.union_find.union(x, y)
    }

    /// Returns the representative of element `i` at index `i`.
    pub fn labels(&self) -> Vec<K> {
        self.union_find.clone().into_labeling()
    }

    /// Returns the representatives of all sets.
    pub fn reprs(&self) -> IndexSet<K> {
        self.labels().into_iter().collect()
    }

    /// Returns a mapping from set representative to set size.
    pub fn counts(&self) -> IndexMap<K, usize> {
        self.to_clustering()
            .into_iter()
            .map(|(repr, set)| (repr, set.len()))
            .collect()
    }

    /// Returns the number of disjoint sets.
    pub fn count(&self) -> usize {
        let mut labels = self.labels();
        labels.sort_unstable();
        labels.dedup();
        labels.len()
    }

    /// Returns a mapping from set representative to its elements.
    pub fn to_clustering(&self) -> Clustering<K> {
        let mut sets: IndexMap<_, IndexSet<_>> = IndexMap::new();
        for node in (0..self.n).map(IndexType::new) {
            let root = self.repr(node);
            sets.entry(root).or_default().insert(node);
        }
        sets
    }
}

#[cfg(test)]
mod tests {
    use petgraph::graph::NodeIndex;

    use super::*;

    fn index(i: usize) -> NodeIndex<usize> {
        NodeIndex::new(i)
    }

    #[test]
    fn a_fresh_structure_holds_only_singletons() {
        let sets: DisjointSets<NodeIndex<usize>> = DisjointSets::new(4);

        assert_eq!(sets.count(), 4);
        assert!(sets.counts().values().all(|&count| count == 1));
    }

    #[test]
    fn union_merges_sets_and_reports_whether_it_changed_anything() {
        let mut sets: DisjointSets<NodeIndex<usize>> = DisjointSets::new(4);

        assert!(sets.union(index(0), index(1)));
        assert!(!sets.union(index(0), index(1)), "already merged");
        assert_eq!(sets.count(), 3);
        assert_eq!(sets.repr(index(0)), sets.repr(index(1)));
    }

    #[test]
    fn clustering_partitions_every_element_exactly_once() {
        let mut sets: DisjointSets<NodeIndex<usize>> = DisjointSets::new(5);
        sets.union(index(0), index(1));
        sets.union(index(1), index(2));
        sets.union(index(3), index(4));

        let clustering = sets.to_clustering();

        assert_eq!(clustering.len(), 2);
        assert_eq!(clustering.values().map(IndexSet::len).sum::<usize>(), 5);
        assert_eq!(sets.reprs().len(), 2);
    }

    #[test]
    fn counts_match_the_clustering_sizes() {
        let mut sets: DisjointSets<NodeIndex<usize>> = DisjointSets::new(5);
        sets.union(index(0), index(1));
        sets.union(index(1), index(2));

        let mut sizes: Vec<_> = sets.counts().into_values().collect();
        sizes.sort_unstable();

        assert_eq!(sizes, vec![1, 1, 3]);
    }

    #[test]
    fn labels_expose_one_representative_per_element() {
        let mut sets: DisjointSets<NodeIndex<usize>> = DisjointSets::new(3);
        sets.union(index(0), index(2));

        let labels = sets.labels();

        assert_eq!(labels.len(), 3);
        assert_eq!(labels[0], labels[2]);
        assert_ne!(labels[0], labels[1]);
    }
}
