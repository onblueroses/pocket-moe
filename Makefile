CC = gcc
CFLAGS = -O3 -Wall -Wextra -std=c17 -D_GNU_SOURCE
# io_uring is optional (Linux-only optimization, blocked on Android via seccomp)
LDFLAGS = -lvulkan -lm -lpthread

# Detect io_uring availability (Linux native builds only, not Android)
HAS_URING := $(shell pkg-config --exists liburing 2>/dev/null && echo 1 || echo 0)
ifeq ($(HAS_URING),1)
  CFLAGS_URING = -DHAS_IO_URING
  LDFLAGS_URING = -luring
endif

# ggml (added as submodule, built separately)
GGML_DIR = ggml
GGML_INC = -I$(GGML_DIR)/include
GGML_LIB = $(GGML_DIR)/build/src/libggml.a

SRC_DIR = src
BUILD_DIR = build

C_SRCS = $(wildcard $(SRC_DIR)/engine/*.c) \
         $(wildcard $(SRC_DIR)/io/*.c) \
         $(wildcard $(SRC_DIR)/model/*.c) \
         $(SRC_DIR)/main.c

C_OBJS = $(C_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)

TARGET = $(BUILD_DIR)/pocket-moe

.PHONY: all clean ggml

all: ggml $(TARGET)

$(TARGET): $(C_OBJS) $(GGML_LIB)
	$(CC) -o $@ $(C_OBJS) $(GGML_LIB) $(LDFLAGS) $(LDFLAGS_URING)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(CFLAGS_URING) $(GGML_INC) -I$(SRC_DIR) -c $< -o $@

ggml:
	@if [ ! -d "$(GGML_DIR)" ]; then \
		echo "Error: ggml submodule not found. Run: git submodule add https://github.com/ggml-org/ggml.git"; \
		exit 1; \
	fi
	@mkdir -p $(GGML_DIR)/build
	cd $(GGML_DIR)/build && cmake .. -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release && make -j$$(nproc)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

# Android cross-compilation
ANDROID_NDK ?= $(NDK_HOME)
ANDROID_API ?= 30
ANDROID_TARGET = aarch64-linux-android$(ANDROID_API)
ANDROID_CC = $(ANDROID_NDK)/toolchains/llvm/prebuilt/linux-x86_64/bin/clang --target=$(ANDROID_TARGET)

android: check-ndk ggml-android
	@mkdir -p $(BUILD_DIR)/android
	$(ANDROID_CC) $(CFLAGS) $(GGML_INC) -I$(SRC_DIR) \
		$(C_SRCS) $(GGML_DIR)/build-android/src/libggml.a \
		-lm -o $(BUILD_DIR)/android/pocket-moe

check-ndk:
	@if [ -z "$(ANDROID_NDK)" ] || [ ! -d "$(ANDROID_NDK)" ]; then \
		echo "Error: ANDROID_NDK or NDK_HOME must point to a valid NDK directory."; \
		echo "Install via: yay -S android-ndk"; \
		exit 1; \
	fi

ggml-android:
	@mkdir -p $(GGML_DIR)/build-android
	cd $(GGML_DIR)/build-android && cmake .. \
		-DCMAKE_TOOLCHAIN_FILE=$(ANDROID_NDK)/build/cmake/android.toolchain.cmake \
		-DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-$(ANDROID_API) \
		-DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release \
		&& make -j$$(nproc)

deploy: android
	adb push $(BUILD_DIR)/android/pocket-moe /data/local/tmp/
	adb push models/ /data/local/tmp/models/
	adb shell chmod +x /data/local/tmp/pocket-moe
