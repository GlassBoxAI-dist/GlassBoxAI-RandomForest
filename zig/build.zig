const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const rf_mod = b.addModule("random_forest", .{
        .root_source_file = b.path("random_forest.zig"),
        .target = target,
        .optimize = optimize,
    });
    rf_mod.addIncludePath(b.path("../include"));
    rf_mod.linkSystemLibrary("facaded_random_forest");
    rf_mod.addLibraryPath(b.path("../target/release"));
    rf_mod.linkLibC();
}
