use std::cmp::min;

/// This struct is used to handle partial buffers, since we have
/// to take buffers as they arrive and there might not be all the
/// data we need.
///
/// The use-case is via the take function, which attempts to grab
/// the amount of data specified, and if it isn't available, stores
/// the partial data in the extra buffer, and returns None.
///
/// Next time around, the extra data will be prepended to the next
/// buffer, so eventually the amount of data requested will become
/// available.
///
/// The concept of reserve is used to handle the case where we need
/// to leave a certain amount of data in the buffer, perticularly
///  where Lepton files have the 32bit file size appended.
/// We don't want this to get parsed out, so we ensure that there are
/// always at least 4 bytes in the buffer.
pub struct PartialBuffer<'a> {
    slice: &'a [u8],
    extra_buffer: &'a mut Vec<u8>,
}

impl<'a> PartialBuffer<'a> {
    pub fn new(slice: &'a [u8], extra_buffer: &'a mut Vec<u8>) -> PartialBuffer<'a> {
        PartialBuffer {
            slice,
            extra_buffer,
        }
    }

    /// grabs a variable amount of data if is available, otherwise put it in the extra
    /// buffer for next time.
    pub fn take(&mut self, size: usize, reserve: usize) -> Option<Vec<u8>> {
        if self.extra_buffer.len() + self.slice.len() < size + reserve {
            self.extra_buffer.extend_from_slice(self.slice);
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

    /// grabs a fixed amount of data if it is available, otherwise put it in the extra
    /// buffer for next time.
    pub fn take_n<const N: usize>(&mut self, reserve: usize) -> Option<[u8; N]> {
        if self.extra_buffer.len() + self.slice.len() < N + reserve {
            self.extra_buffer.extend_from_slice(self.slice);
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
