add_subdirectory(Runtime)
add_subdirectory(Support)
# Accelerators introduces a target AcceleratorsInc. Define a dummy one here
add_custom_target(AcceleratorsInc
    COMMAND echo "This is the dummy definition for AcceleratorsInc"
)
