CC = clang
OBJC = clang
CFLAGS = -O3 -Wall -Wextra -std=c17
OBJCFLAGS = -O3 -Wall -Wextra -fobjc-arc
LDFLAGS = -framework Metal -framework MetalPerformanceShaders \
          -framework Accelerate -framework Foundation

METAL_CC = xcrun -sdk iphoneos metal
METAL_FLAGS = -std=metal3.1

SRC_DIR = src
BUILD_DIR = build

# Source files
C_SRCS = $(wildcard $(SRC_DIR)/engine/*.c) \
         $(wildcard $(SRC_DIR)/io/*.c) \
         $(wildcard $(SRC_DIR)/model/*.c)
OBJC_SRCS = $(wildcard $(SRC_DIR)/*.m)
METAL_SRCS = $(wildcard $(SRC_DIR)/metal/*.metal)

C_OBJS = $(C_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
OBJC_OBJS = $(OBJC_SRCS:$(SRC_DIR)/%.m=$(BUILD_DIR)/%.o)
METAL_LIBS = $(METAL_SRCS:$(SRC_DIR)/%.metal=$(BUILD_DIR)/%.metallib)

TARGET = $(BUILD_DIR)/pocket-moe

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(C_OBJS) $(OBJC_OBJS) $(METAL_LIBS)
	$(OBJC) $(LDFLAGS) -o $@ $(C_OBJS) $(OBJC_OBJS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -I$(SRC_DIR) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.m | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(OBJC) $(OBJCFLAGS) -I$(SRC_DIR) -c $< -o $@

$(BUILD_DIR)/%.metallib: $(SRC_DIR)/%.metal | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(METAL_CC) $(METAL_FLAGS) -o $(BUILD_DIR)/$*.air $<
	xcrun -sdk iphoneos metallib -o $@ $(BUILD_DIR)/$*.air

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
