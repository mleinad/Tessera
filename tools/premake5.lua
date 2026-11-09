-- tools/premake5.lua - Tessera
-- Run from the repository root:  premake5 --file=tools/premake5.lua xcode4

-- Repository root = the parent of the folder this script lives in
local ROOT = path.getabsolute(path.join(_SCRIPT_DIR, ".."))

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
    cppdialect "C++20"                
    location  (ROOT .. "/build")
    targetdir (ROOT .. "/Bin/%{cfg.buildcfg}")
    objdir    (ROOT .. "/Bin/Intermediates/%{cfg.buildcfg}")

    files {
        ROOT .. "/src/main.cpp",
        ROOT .. "/src/common/**.h",
        ROOT .. "/src/common/**.hpp",
        ROOT .. "/src/common/**.cpp",
    }
    includedirs { ROOT .. "/src", ROOT .. "/src/common" }

    filter "configurations:Debug"
        defines { "TESSERA_DEBUG" }
        symbols "On"
        optimize "Off"

    filter "configurations:Release"
        defines { "NDEBUG" }                -- disables assert(): keep side effects out of asserts
        optimize "Speed"
        symbols "On"

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
            ROOT .. "/src/sim/**.hpp",      -- still Metal-specific until abstracted
            ROOT .. "/src/sim/**.cpp",
        }

        removefiles { ROOT .. "/src/backends/metal/metal-cpp/**" }

        includedirs { ROOT .. "/src/backends/metal/metal-cpp" }
        links { "Metal.framework", "Foundation.framework", "QuartzCore.framework" }

        xcodebuildsettings {
            ["MTL_HEADER_SEARCH_PATHS"] = ROOT .. "/src/common",
            ["MTL_FAST_MATH"]           = "NO",
        }

        -- kernel.metal -> kernel.air -> default.metallib next to the executable
        postbuildcommands {
            '{MKDIR} "%{cfg.objdir}"',
            'xcrun -sdk macosx metal -c -fno-fast-math'
                .. ' -I "' .. ROOT .. '/src/common"'
                .. ' "' .. ROOT .. '/src/backends/metal/kernel.metal"'
                .. ' -o "%{cfg.objdir}/kernel.air"',
            'xcrun -sdk macosx metallib "%{cfg.objdir}/kernel.air"'
                .. ' -o "%{cfg.targetdir}/default.metallib"',
        }

    filter "system:windows or linux"
        defines { "TESSERA_BACKEND_CUDA" }
        files {
            ROOT .. "/src/backends/cuda/**.hpp",
            ROOT .. "/src/backends/cuda/**.cpp",
            ROOT .. "/src/backends/cuda/**.cu",
        }

    filter {}