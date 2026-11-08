local ROOT = path.getabsolute(".")

workspace "Tessera"
    configurations { "Debug", "Release" }
    location "build"
    startproject "Tessera"

    filter "system:macosx"
        architecture "ARM64"
    filter "system:windows or linux"
        architecture "x86_64"
    filter {}

project "Tessera"
    kind "ConsoleApp"
    language "C++"
    cppdialect "C++17"                      -- metal-cpp requires C++17
    targetdir (ROOT .. "/bin/%{cfg.system}-%{cfg.buildcfg}")
    objdir    (ROOT .. "/bin-int/%{cfg.system}-%{cfg.buildcfg}")

    ---------------------------------------------------------------------------
    -- Shared by every platform
    ---------------------------------------------------------------------------
    files {
        "src/main.cpp",
        "src/common/**.h",
        "src/common/**.hpp",
        "src/common/**.cpp",
    }
    includedirs { "src", "src/common" }

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
            "src/backends/metal/**.hpp",
            "src/backends/metal/**.cpp",
            "src/backends/metal/**.metal",
            -- The simulator still uses Metal types directly. Move it to the shared
            -- list above once it talks to a backend interface instead.
            "src/sim/**.hpp",
            "src/sim/**.cpp",
        }
        includedirs { "external/metal-cpp" }
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

    filter "system:windows or linux"
        defines { "TESSERA_BACKEND_CUDA" }
        files {
            "src/backends/cuda/**.hpp",
            "src/backends/cuda/**.cpp",
            "src/backends/cuda/**.cu",
        }
 
    filter {}