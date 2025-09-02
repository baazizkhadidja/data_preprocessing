#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 09:17:55 2025

@author: khadidjabaaziz
"""

from torch.utils.data import DataLoader

dataset = ["pomme", "banane", "orange", "kiwi"]
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch in dataloader:
    print(batch)
    
    
batch=next(iter(dataloader))
batch
