use std::cmp::min;

/// This struct is used to the fact that we have to take buffers
/// as they arrive, and we might not have all the data we need.
///
/// This used is via the take function, which attempts to grab
/// the amount of data specified, and if it isn't available, stores
/// the partial data in the extra buffer, and returns None.
///
/// Next time around, the extra data will be prepended to the next
/// buffer, so eventually the amount of data requested will become
/// available.
///
/// The concept of retention_bytes is used to handle the case where we need
/// to leave a certain amount of data in the buffer, perticularly
/// where Lepton files have the 32bit file size appended.
///
/// We don't want this to get parsed out, so we ensure that there are
/// always at least 4 bytes in the buffer.
pub struct PartialBuffer<'a> {
    slice: &'a [u8],
    extra_buffer: &'a mut Vec<u8>,
    continue_processing: bool,
}

impl<'a> PartialBuffer<'a> {
    /// Instantiates a new buffer with a slice and a place to store extra data
    /// between calls.
    ///
    /// Extra data is used both to remember extra data from the previous buffer
    /// and is updated with any data that is left over after a take call.
    pub fn new(slice: &'a [u8], extra_buffer: &'a mut Vec<u8>) -> PartialBuffer<'a> {
        PartialBuffer {
            slice,
            extra_buffer,
            continue_processing: true,
        }
    }

    /// returns true if we haven't yet run out of data (ie take returned empty)
    pub fn continue_processing(&self) -> bool {
        self.continue_processing
    }

    /// Attempts to get "size" bytes of data from the buffer. If that much
    /// is available (including the extra buffer from the previous call), it is
    /// returned as a vector exactly that size, otherwise the data is appended
    /// to the extra buffer and None is returned.
    ///
    /// retention_bytes (see comment at top of file) indicates that we should never
    /// consume the last x bytes of the buffer. This is useful because of the particular
    /// way that Lepton files are encoded, the file size is appended without any sort
    /// of header or marker, so the only way to know we are at the end is if there
    /// are only 4 bytes left.
    pub fn take(&mut self, size: usize, retention_bytes: usize) -> Option<Vec<u8>> {
        if self.extra_buffer.len() + self.slice.len() < size + retention_bytes {
            self.extra_buffer.extend_from_slice(self.slice);
            self.slice = &[];
            self.continue_processing = false;
            return None;
        }

        let mut retval = Vec::with_capacity(size);
        let amount_from_extra = min(self.extra_buffer.len(), size);
        if amount_from_extra > 0 {
            retval.extend_from_slice(&self.extra_buffer[0..amount_from_extra]);
            self.extra_buffer.drain(0..amount_from_extra);
        }

        let amount_from_slice = size - amount_from_extra;
        if amount_from_slice > 0 {
            retval.extend_from_slice(&self.slice[0..amount_from_slice]);
            self.slice = &self.slice[amount_from_slice..];
        }

        debug_assert!(retval.len() == size);
        return Some(retval);
    }

    /// Same as take, except returns a fixed size array instead of a vector.
    ///
    /// Useful when we are expecting a small fixed number of bytes like a header
    /// or signature.
    pub fn take_n<const N: usize>(&mut self, retention_bytes: usize) -> Option<[u8; N]> {
        if self.extra_buffer.len() + self.slice.len() < N + retention_bytes {
            self.extra_buffer.extend_from_slice(self.slice);
            self.slice = &[];
            self.continue_processing = false;
            return None;
        }

        let mut retval = [0; N];
        let amount_from_extra = min(self.extra_buffer.len(), N);
        if amount_from_extra > 0 {
            retval[0..amount_from_extra].copy_from_slice(&self.extra_buffer[0..amount_from_extra]);
            self.extra_buffer.drain(0..amount_from_extra);
        }

        let amount_from_slice = N - amount_from_extra;
        if amount_from_slice > 0 {
            retval[amount_from_extra..N].copy_from_slice(&self.slice[0..amount_from_slice]);
            self.slice = &self.slice[amount_from_slice..];
        }

        Some(retval)
    }
}

#[test]
fn test_taking_simple() {
    let mut extra = Vec::new();
    let mut pb = PartialBuffer::new(&[1, 2, 3, 4], &mut extra);

    let taken = pb.take(4, 0).unwrap();
    assert_eq!(taken, vec![1, 2, 3, 4]);
    assert_eq!(&extra[..], []);
}

#[test]
fn test_taking_simple_n() {
    let mut extra = Vec::new();
    let mut pb = PartialBuffer::new(&[1, 2, 3, 4], &mut extra);

    let taken = pb.take_n::<4>(0).unwrap();
    assert_eq!(taken, [1, 2, 3, 4]);
    assert_eq!(&extra[..], []);
}

#[test]
fn test_taking_extra() {
    let mut extra = Vec::new();
    let mut pb = PartialBuffer::new(&[1, 2, 3, 4], &mut extra);

    // try to take 5 characters, but there are only 4, so it should return None and
    // leave the data read in extra
    assert_eq!(pb.take(5, 0), None);
    assert_eq!(&extra, &vec![1, 2, 3, 4]);

    // now we should be able to take the 4 characters
    let mut pb = PartialBuffer::new(&[5, 6, 7, 8], &mut extra);

    assert_eq!(pb.take(5, 0), Some(vec![1, 2, 3, 4, 5]));

    // try to take another 5, but there aren't
    assert_eq!(pb.take(5, 0), None);

    // the 3 characters we couldn't get should be in extra
    assert!(!pb.continue_processing());
    assert_eq!(&extra, &vec![6, 7, 8]);
}

#[test]
fn test_taking_extra_n() {
    let mut extra = Vec::new();
    let mut pb = PartialBuffer::new(&[1, 2, 3, 4], &mut extra);

    // try to take 5 characters, but there are only 4, so it should return None and
    // leave the data read in extra
    assert_eq!(pb.take_n::<5>(0), None);
    assert_eq!(&extra, &vec![1, 2, 3, 4]);

    // now we should be able to take the 4 characters
    let mut pb = PartialBuffer::new(&[5, 6, 7, 8], &mut extra);

    assert_eq!(pb.take_n::<5>(0), Some([1, 2, 3, 4, 5]));

    // try to take another 5, but there aren't
    assert_eq!(pb.take_n::<5>(0), None);

    // the 3 characters we couldn't get should be in extra
    assert!(!pb.continue_processing());
    assert_eq!(&extra, &vec![6, 7, 8]);
}

#[test]
fn test_taking_reserve() {
    let mut extra = Vec::new();
    let mut pb = PartialBuffer::new(&[1, 2, 3, 4, 5], &mut extra);

    // taking 5 should fail because we wanted a reserve
    assert_eq!(pb.take(5, 1), None);

    let mut pb = PartialBuffer::new(&[], &mut extra);
    assert_eq!(pb.take(5, 0), Some(vec![1, 2, 3, 4, 5]));
}
