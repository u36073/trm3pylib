import collections

MemoryUnit = collections.namedtuple('MemoryUnit', ['bytes', 'name', 'abbreviation'])
MemoryUnits = {'B': MemoryUnit(1, 'Byte', 'B'),
               'KB': MemoryUnit(1000, 'Kilobyte', 'KB'),
               'KIB': MemoryUnit(1024, 'Kibibyte', 'KiB)'),
               'MB': MemoryUnit(1000000, 'Megabyte', 'MB'),
               'MIB': MemoryUnit(1048576, 'Mebibyte', 'MiB'),
               'GB': MemoryUnit(1000000000, 'Gigabyte', 'GB'),
               'GIB': MemoryUnit(1073741824, 'Gibibyte', 'GiB'),
               'TB': MemoryUnit(1000000000000, 'Terabyte', 'TB'),
               'TIB': MemoryUnit(1099511627776, 'Tebibyte', 'TiB'),
               'PB': MemoryUnit(1000000000000000, 'Petabyte', 'PB'),
               'PIB': MemoryUnit(1125899906842624, 'Pebibyte', 'PiB'),
               'EB': MemoryUnit(1000000000000000000, 'Exabyte', 'EB'),
               'EIB': MemoryUnit(1152921504606846976, 'Exbibyte', 'EiB'),
               'ZB': MemoryUnit(1000000000000000000000, 'Zettabyte', 'ZB'),
               'ZiB': MemoryUnit(1180591620717411303424, 'Zebibyte', 'ZiB')
               }
