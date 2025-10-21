use std::{collections::VecDeque, io::Read, sync::mpsc::{channel, Receiver, Sender}};

use crate::{LeptonError, Result};

pub struct IncrementalRead
{
    current_block: VecDeque<u8>,
    partitions : Vec<Partitions>,
}

impl Default for IncrementalRead {
    fn default() -> Self {
        IncrementalRead {
            current_block: VecDeque::new(),
            partitions: Vec::new(),
        }
    }
}

struct Partitions
{
    amount_left: usize,
    receiver: Receiver<Vec<u8>>,
}

impl IncrementalRead {
    pub fn append_content(&mut self, data: &[u8]) {
        self.current_block.extend(data);
    }

    pub fn add_partition(&mut self, amount: usize) -> Sender<Vec<u8>> {
        let (sender, receiver) = channel();

        self.partitions.push(Partitions {
            amount_left: amount,
            receiver,
        });

        sender
    }
}

impl Read for IncrementalRead {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {

        while self.current_block.is_empty() && !self.partitions.is_empty()
        {
            let partition = &mut self.partitions[0];
            if partition.amount_left == 0 {
                self.partitions.remove(0);
                continue;
            }

            match partition.receiver.recv() {
                Ok(mut block) => {
                    if partition.amount_left < block.len() {
                        // shorten block if too much data was sent
                        log::warn!("Received more data than expected from worker, truncating");
                        block.truncate(partition.amount_left);
                    }

                    partition.amount_left -= block.len();

                    self.current_block = VecDeque::from(block);
                    break;
                }
                Err(e) => {
                    return Err(LeptonError::from(e).into());
                }
            }
        }
        return self.current_block.read(buf);
    }
}