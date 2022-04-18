import pkg_resources
import sys

f = open(sys.argv[1])
lines = f.readlines()
covered_pkgs = set()
for fname in sys.argv[2:]:
    f = open(fname)
    lines2 = f.readlines()
    for line in lines2:
        covered_pkgs.add(line.split('=')[0].split('@')[0].strip())
for line in lines:
    mlines = None
    try:
        pkg_name = line.split('=')[0].split('@')[0].strip()
        if pkg_name in covered_pkgs:
            continue
        pkg = pkg_resources.require(pkg_name)[0]
        mlines = pkg.get_metadata_lines('PKG-INFO')
    except:
        try:
            mlines = pkg.get_metadata_lines('METADATA')
        except:
            pass
    license = "Unknown"
    if mlines is not None:
        for mline in mlines:
            if 'License:' in mline:
                license = mline.split('License:')[1]
                break
    print(pkg_name+','+license)
