import sys
sys.path.insert(0, './target/debug')

import libeig_crate
print(libeig_crate.sum_as_string(5, 20))
