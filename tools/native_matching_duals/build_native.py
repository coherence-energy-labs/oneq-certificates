"""Build the isolated native-dual bridge; does not patch/install PyMatching."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ap=argparse.ArgumentParser()
ap.add_argument("--upstream",type=Path,required=True)
ap.add_argument("--out",type=Path,required=True)
a=ap.parse_args()
root=Path(__file__).resolve().parent
sources=[]
for sub in ("flooder","flooder_matcher_interop","matcher","search","tracker"):
    sources += sorted((a.upstream/"src/pymatching/sparse_blossom"/sub).glob("*.cc"))
sources=[p for p in sources if not any(x in p.name for x in (".test.",".perf."))]
runtime_flags=[] if os.name=="nt" else ["-fPIC"]
cmd=["g++","-std=c++20","-O3","-DNDEBUG",*runtime_flags,"-shared","-I",str(a.upstream/"src"),str(root/"native_dual.cc"),*[str(p) for p in sources],"-o",str(a.out)]
a.out.parent.mkdir(parents=True,exist_ok=True)
subprocess.run(cmd,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
runtime_dlls={}
if os.name=="nt":
    # Modern ctypes searches the DLL's directory, not arbitrary PATH entries.
    # Keep the compiler's own MinGW runtimes beside the generated DLL and bind
    # their bytes in the build receipt. No machine-wide DLL search change.
    compiler_bin=Path(shutil.which("g++")).resolve().parent
    for name in ("libstdc++-6.dll","libgcc_s_seh-1.dll","libwinpthread-1.dll"):
        dependency=compiler_bin/name
        if dependency.is_file():
            target=a.out.parent/name
            if dependency.resolve()!=target.resolve():
                shutil.copy2(dependency,target)
            runtime_dlls[name]=sha(target)
headers=sorted(p for p in (a.upstream/"src/pymatching/sparse_blossom").rglob("*") if p.suffix in (".h",".inl") and ".test." not in p.name)
manifest={"upstream":"https://github.com/oscarhiggott/PyMatching","upstream_commit":subprocess.check_output(["git","-C",str(a.upstream),"rev-parse","HEAD"],text=True).strip(),"compiler":subprocess.check_output(["g++","--version"],text=True).splitlines()[0],"command":cmd,"source_sha256":{str(p.relative_to(a.upstream)):sha(p) for p in sources+headers},"bridge_sha256":sha(root/"native_dual.cc"),"build_script_sha256":sha(Path(__file__)),"binary_sha256":sha(a.out),"upstream_source_modified":bool(subprocess.check_output(["git","-C",str(a.upstream),"status","--porcelain"],text=True).strip())}
manifest["bundled_runtime_dll_sha256"]=runtime_dlls
a.out.with_suffix(".manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"compiled":str(a.out),"source_count":len(sources),"upstream_commit":manifest["upstream_commit"],"upstream_source_modified":manifest["upstream_source_modified"]}))
