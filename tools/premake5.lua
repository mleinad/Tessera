-- Repo root = parent of tools/. Plain relative paths would resolve from tools/.
local ROOT = path.getabsolute("..", _SCRIPT_DIR)

workspace "Tessera"
    configurations { "Debug", "Release" }
    location (ROOT .. "/build")
    startproject "Tessera"

    filter "system:macosx"
        architecture "ARM64"
    filter "system:windows or linux"
        architecture "x86_64"
    filter {}

project "Tessera"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++20"                      -- logger.hpp uses <format>; metal-cpp needs >= C++17
    targetdir (ROOT .. "/bin/%{cfg.system}-%{cfg.buildcfg}")
    objdir    (ROOT .. "/bin-int/%{cfg.system}-%{cfg.buildcfg}")

    ---------------------------------------------------------------------------
    -- Shared by every platform
    ---------------------------------------------------------------------------
    files {
        ROOT .. "/src/main.cpp",
        ROOT .. "/src/common/**.h",
        ROOT .. "/src/common/**.hpp",
        ROOT .. "/src/common/**.cpp",
        ROOT .. "/src/sim/**.hpp",
        ROOT .. "/src/sim/**.cpp",
    }
    includedirs { ROOT .. "/src", ROOT .. "/src/common" }

    filter "configurations:Debug"
        defines { "TESSERA_DEBUG" }
        symbols "On"
        optimize "Off"

    filter "configurations:Release"
        defines { "NDEBUG" }                -- disables assert(): keep side effects out of asserts
        optimize "Speed"
        symbols "On"                        -- keep symbols so profilers can show function names

    -- Warn when a local variable shadows a member (the bug that bit the Engine and KernelLibrary)
    filter "toolset:clang or gcc"
        buildoptions { "-Wshadow" }
    filter "toolset:msc*"
        buildoptions { "/w14456", "/w14457", "/w14458", "/w14459" }

    filter "system:macosx"
        defines { "TESSERA_BACKEND_METAL" }
        files {
            ROOT .. "/src/backends/metal/**.hpp",
            ROOT .. "/src/backends/metal/**.cpp",
            ROOT .. "/src/backends/metal/**.metal",
        }
        includedirs { ROOT .. "/src/backends/metal/metal-cpp" }
        links { "Metal.framework", "Foundation.framework", "QuartzCore.framework" }

        -- Used if Xcode compiles the .metal file itself, so both paths build the same library
        xcodebuildsettings {
            ["MTL_HEADER_SEARCH_PATHS"] = ROOT .. "/src/common",
            ["MTL_FAST_MATH"]           = "NO",
        }

        -- Explicit shader build, identical for Xcode and make:
        -- kernel.metal -> kernel.air -> default.metallib next to the executable,
        -- where newDefaultLibrary() looks for it. Fast math off to match CUDA.
        postbuildcommands {
            '{MKDIR} "%{cfg.objdir}"',
            'xcrun -sdk macosx metal -c -fno-fast-math'
                .. ' -I "' .. ROOT .. '/src/common"'
                .. ' "' .. ROOT .. '/src/backends/metal/kernel.metal"'
                .. ' -o "%{cfg.objdir}/kernel.air"',
            'xcrun -sdk macosx metallib "%{cfg.objdir}/kernel.air"'
                .. ' -o "%{cfg.targetdir}/default.metallib"',
        }

    -- Linux: CUDA backend
    -- nvcc from PATH (apt install puts it in /usr/bin) unless CUDA_PATH is set
    local CUDA = os.getenv("CUDA_PATH")

    -- Common nvcc command. sm_86 = RTX 30-series (Ampere, incl. the 3070 Ti Laptop).
    local NVCC = (CUDA and ('"' .. CUDA .. '/bin/nvcc"') or "nvcc")
              .. ' -std=c++20 -arch=sm_86'
              .. ' -I"' .. ROOT .. '/src" -I"' .. ROOT .. '/src/common"'
              .. ' -DTESSERA_BACKEND_CUDA'
    local CU_OUT = '"%{cfg.objdir}/%{file.basename}.cu.o"'

    filter "system:linux"
        defines { "TESSERA_BACKEND_CUDA" }
        files {
            ROOT .. "/src/backends/cuda/**.hpp",
            ROOT .. "/src/backends/cuda/**.cpp",
            ROOT .. "/src/backends/cuda/**.cu",
        }
        links { "cudart" }
        if CUDA then  -- apt CUDA uses system include/lib dirs; only add these for a custom install
            includedirs { CUDA .. "/include" }                 -- cuda_runtime.h for .cpp files
            libdirs     { CUDA .. "/lib64" }
            linkoptions { "-Wl,-rpath," .. CUDA .. "/lib64" }  -- find libcudart.so at run time
        end

    -- Every .cu file goes through nvcc; the .o it produces is linked automatically
    filter { "system:linux", "files:**.cu" }
        buildmessage "nvcc %{file.name}"
        buildinputs  { ROOT .. "/src/common/WaveCommon.h" }   -- rebuild when the shared kernel header changes
        buildoutputs { "%{cfg.objdir}/%{file.basename}.cu.o" }

    -- Debug: -G enables stepping through kernels in cuda-gdb (kernels run much slower)
    filter { "system:linux", "files:**.cu", "configurations:Debug" }
        buildcommands {
            '{MKDIR} "%{cfg.objdir}"',
            NVCC .. ' -DTESSERA_DEBUG -G -g -O0 -c "%{file.abspath}" -o ' .. CU_OUT,
        }

    -- Release: full speed; -lineinfo keeps source lines for Nsight profilers
    filter { "system:linux", "files:**.cu", "configurations:Release" }
        buildcommands {
            '{MKDIR} "%{cfg.objdir}"',
            NVCC .. ' -DNDEBUG -O3 -lineinfo -c "%{file.abspath}" -o ' .. CU_OUT,
        }

    filter {}