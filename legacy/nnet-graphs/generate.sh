#!/usr/bin/env bash

# KUHN2
# python qnet.py PREDICT  p 7 3 '[16]'
# python qnet.py TRAIN    p 7 3 '[16]'
# python qnet.py PREDICT  q 7 3 '[16]'
# python qnet.py TRAIN    q 7 3 '[16]'

# KUHN3
# python qnet.py PREDICT  p 10 3 '[16]'
# python qnet.py TRAIN    p 10 3 '[16]'
# python qnet.py PREDICT  q 10 3 '[16]'
# python qnet.py TRAIN    q 10 3 '[16]'

# LEDUC2
# python qnet.py PREDICT  p 13 3 '[32]'
# python qnet.py TRAIN    p 13 3 '[32]'
# python qnet.py PREDICT  q 13 3 '[32]'
# python qnet.py TRAIN    q 13 3 '[32]'

# LEDUC2 - FRENCH
# python qnet.py PREDICT  p 25 3 '[1024, 512]'
# python qnet.py TRAIN    p 25 3 '[1024, 512]'
# python qnet.py PREDICT  q 25 3 '[1024, 512]'
# python qnet.py TRAIN    q 25 3 '[1024, 512]'

# COCHARD2
# python qnet.py PREDICT  p 19 3 '[32]'
# python qnet.py TRAIN    p 19 3 '[32]'
# python qnet.py PREDICT  q 19 3 '[32]'
# python qnet.py TRAIN    q 19 3 '[32]'

# TEXAS LIMIT
# python qnet.py PREDICT  p 28 3 '[1024]'
# python qnet.py TRAIN    p 28 3 '[1024]'
# python qnet.py PREDICT  q 28 3 '[1024]'
# python qnet.py TRAIN    q 28 3 '[1024]'

# TEXAS LIMIT ZERO 2
# python qnet.py PREDICT  p 304 3 '[1024, 512]'
# python qnet.py TRAIN    p 304 3 '[1024, 512]'
# python qnet.py PREDICT  q 304 3 '[1024, 512]'
# python qnet.py TRAIN    q 304 3 '[1024, 512]'

# TEXAS LIMIT ZERO 3
python qnet.py PREDICT  p 352 3 '[1024, 512]'
python qnet.py TRAIN    p 352 3 '[1024, 512]'
python qnet.py PREDICT  q 352 3 '[1024, 512]'
python qnet.py TRAIN    q 352 3 '[1024, 512]'

mv qnet*.pb* ../resources/graphs/

# SNET
# python snet.py TRAIN    104 '[512]'
# python snet.py PREDICT  104 '[512]'

mv snet*.pb* ../resources/graphs/

