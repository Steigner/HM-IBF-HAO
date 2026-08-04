//! The backbone geometry a working dimension's control points sit on.

/// The positions, normals and offset bounds of one working dimension's control points.
///
/// The three arrays are always the same length - the working dimension - and are indexed
/// together: entry `i` describes the control point offset `i` displaces.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BackboneSamples {
    /// The control positions, in heightmap coordinates.
    pub positions: Vec<[f64; 2]>,
    /// The unit normal at each position.
    pub normals: Vec<[f64; 2]>,
    /// The inclusive `(min, max)` offset bounds at each position.
    pub bounds: Vec<(f64, f64)>,
}

impl BackboneSamples {
    /// Creates empty samples with room for `dimension` control points.
    ///
    /// # Arguments
    ///
    /// * `dimension` - The working dimension to reserve capacity for.
    ///
    /// # Returns
    ///
    /// Empty samples.
    pub fn with_capacity(dimension: usize) -> Self {
        Self {
            positions: Vec::with_capacity(dimension),
            normals: Vec::with_capacity(dimension),
            bounds: Vec::with_capacity(dimension),
        }
    }

    /// Appends one control point.
    ///
    /// # Arguments
    ///
    /// * `position` - The control position.
    /// * `normal` - Its unit normal.
    /// * `bounds` - Its inclusive offset bounds.
    pub fn push(&mut self, position: [f64; 2], normal: [f64; 2], bounds: (f64, f64)) {
        self.positions.push(position);
        self.normals.push(normal);
        self.bounds.push(bounds);
    }

    /// Returns the number of control points.
    ///
    /// # Returns
    ///
    /// The working dimension these samples describe.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns whether there are no control points at all.
    ///
    /// # Returns
    ///
    /// `true` for a working dimension of zero.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_samples_are_empty() {
        let samples = BackboneSamples::with_capacity(8);

        assert_eq!(samples.len(), 0);
        assert!(samples.is_empty());
    }

    #[test]
    fn pushing_keeps_every_array_aligned() {
        let mut samples = BackboneSamples::with_capacity(2);

        samples.push([1.0, 2.0], [0.0, 1.0], (-1.0, 1.0));
        samples.push([3.0, 4.0], [1.0, 0.0], (-2.0, 2.0));

        assert_eq!(samples.len(), 2);
        assert_eq!(samples.normals.len(), samples.len());
        assert_eq!(samples.bounds.len(), samples.len());
        assert_eq!(samples.positions[1], [3.0, 4.0]);
    }
}
