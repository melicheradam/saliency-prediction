#!/bin/bash

rm -rf encoder-decoder-model/results/ckpts/best
rm -rf encoder-decoder-model/results/ckpts/latest
cp -r encoder-decoder-model/results/ckpts/$1-best/best encoder-decoder-model/results/ckpts/
cp -r encoder-decoder-model/results/ckpts/$1-latest/latest encoder-decoder-model/results/ckpts/

rm -rf encoder-decoder-model/results/history
mkdir encoder-decoder-model/results/history
#cp -r encoder-decoder-model/results/$1-history/history encoder-decoder-model/results/
