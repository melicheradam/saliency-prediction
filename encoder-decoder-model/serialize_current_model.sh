#!/bin/bash

rm -rf encoder-decoder-model/results/ckpts/$1-best
rm -rf encoder-decoder-model/results/ckpts/$1-latest
rm -rf encoder-decoder-model/results/$1-history

mkdir encoder-decoder-model/results/ckpts/$1-best
mkdir encoder-decoder-model/results/ckpts/$1-latest
cp -r encoder-decoder-model/results/ckpts/best encoder-decoder-model/results/ckpts/$1-best
cp -r encoder-decoder-model/results/ckpts/latest encoder-decoder-model/results/ckpts/$1-latest

mkdir encoder-decoder-model/results/$1-history
cp -r encoder-decoder-model/results/history encoder-decoder-model/results/$1-history

# Remove history because training depends on it (mean history error)
rm -rf encoder-decoder-model/results/history
mkdir encoder-decoder-model/results/history



