#!/usr/bin/env python3

from gnuradio import gr, blocks, digital, fec, channels
import numpy as np

class UTF8LDPCCommSystem(gr.top_block):
    def __init__(self, text, matrix_file, max_iter=50, eb_no_db=3.0):
        gr.top_block.__init__(self, "UTF-8 LDPC Comm System")

        # === PARAMETERS ===
        self.bits_per_symbol = 1  # BPSK
        self.samp_rate = 100
        self.text = text
        # Convert Eb/N0 (dB) to noise variance
        eb_no_linear = 10**(eb_no_db/10.0)
        snr_linear = eb_no_linear * self.bits_per_symbol
        noise_variance = 1.0 / snr_linear

        # === SOURCE & UTF-8 ENCODING ===
        byte_array = bytearray(self.text, 'utf-8')
        self.src = blocks.vector_source_b(list(byte_array), repeat=False)
        self.unpack = blocks.unpack_k_bits_bb(8)

        # === LDPC ENCODER ===
        # Load parity-check matrix from AList file
        self.ldpc_H = fec.ldpc_H_matrix(matrix_file, 3)
        print('i was herr')

        self.ldpc_enc = fec.encoder(
            fec.ldpc_par_mtrx_encoder_make(self.ldpc_H),
            False, 0
        )

        # === MAPPING & CHANNEL ===
        # Map bits [0,1] -> BPSK symbols [+1, -1]
        self.chunks_to_syms = digital.chunks_to_symbols_bc((1.0, -1.0), 1)
        self.channel = channels.channel_model(
            noise_voltage=np.sqrt(noise_variance),
            frequency_offset=0.0,
            epsilon=1.0,
            taps=[1.0],
            noise_seed=0,
            block_tags=False
        )

        # === RECEIVER: DEMOD -> LDPC DECODER -> PACK ===
        # BPSK demod: hard decision slicing
        self.demod = digital.binary_slicer_fb()
        # Soft-decision LDPC decoder with belief propagation
        self.ldpc_dec = fec.decoder(
            fec.ldpc_decoder_make(matrix_file, max_iter, 3),
            False, 0
        )
        self.pack = blocks.pack_k_bits_bb(8)
        self.snk = blocks.vector_sink_b()

        # === CONNECT FLOWGRAPH ===
        self.connect(self.src, self.unpack)
        self.connect(self.unpack, self.ldpc_enc)
        self.connect(self.ldpc_enc, self.chunks_to_syms)
        self.connect(self.chunks_to_syms, self.channel)
        self.connect(self.channel, self.demod)
        self.connect(self.demod, self.ldpc_dec)
        self.connect(self.ldpc_dec, self.pack)
        self.connect(self.pack, self.snk)

    def get_received_text(self):
        data = bytes(self.snk.data())
        return data.decode('utf-8', errors='replace')

if __name__ == '__main__':
    tx_text = "Hello, LDPC in GNU Radio!"
    # Path to an AList-format parity-check matrix
    matrix_file = './ldpc_matrix.alist'
    tb = UTF8LDPCCommSystem(tx_text, matrix_file, max_iter=20, eb_no_db=5.0)

    tb.run()
    rx = tb.get_received_text()
    print("Transmitted:", tx_text)
    print("Received   :", rx)
