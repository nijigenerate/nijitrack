#pragma once
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
#endif

namespace niji::platform {
#ifdef _WIN32
using Socket = SOCKET;
inline constexpr Socket invalid_socket = INVALID_SOCKET;
inline void close_socket(Socket fd) {
  closesocket(fd);
}
class Network {
public:
  Network() {
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
      throw std::runtime_error("WSAStartup failed");
  }
  ~Network() {
    WSACleanup();
  }
};
#else
using Socket = int;
inline constexpr Socket invalid_socket = -1;
inline void close_socket(Socket fd) {
  ::close(fd);
}
class Network {};
#endif
inline auto send_packet(Socket fd, const std::vector<uint8_t> &data) {
  return ::send(fd, reinterpret_cast<const char *>(data.data()), static_cast<int>(data.size()), 0);
}

// Spawn a child process with an argument vector; used by process handling tests.
class PipeProcess {
  FILE *stream_ = nullptr;
#ifdef _WIN32
  HANDLE process_ = nullptr;
  static std::wstring wide(const std::string &text) {
    int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.c_str(), -1, nullptr, 0);
    if (!count)
      throw std::runtime_error("Invalid UTF-8 process argument");
    std::wstring result(static_cast<size_t>(count), L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.c_str(), -1, result.data(), count);
    result.pop_back();
    return result;
  }
  static std::wstring argument(const std::string &text) {
    std::wstring out = L"\"";
    size_t slashes = 0;
    for (wchar_t c : wide(text)) {
      if (c == L'\\') {
        ++slashes;
        continue;
      }
      out.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
      out += c;
      slashes = 0;
    }
    out.append(slashes * 2, L'\\');
    return out + L'"';
  }
#else
  pid_t process_ = -1;
#endif
public:
  explicit PipeProcess(std::vector<std::string> args) {
#ifdef _WIN32
    SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
    HANDLE read_pipe = nullptr, write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &security, 0))
      throw std::runtime_error("CreatePipe failed");
    if (!SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0)) {
      CloseHandle(read_pipe);
      CloseHandle(write_pipe);
      throw std::runtime_error("SetHandleInformation failed");
    }
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdOutput = write_pipe;
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    PROCESS_INFORMATION child{};
    std::wstring command;
    for (const auto &arg : args) {
      if (!command.empty())
        command += L' ';
      command += argument(arg);
    }
    BOOL ok = CreateProcessW(nullptr, command.data(), nullptr, nullptr, TRUE, CREATE_NO_WINDOW,
                             nullptr, nullptr, &startup, &child);
    CloseHandle(write_pipe);
    if (!ok) {
      CloseHandle(read_pipe);
      throw std::runtime_error("Cannot start child process");
    }
    CloseHandle(child.hThread);
    process_ = child.hProcess;
    int fd = _open_osfhandle(reinterpret_cast<intptr_t>(read_pipe), _O_RDONLY | _O_BINARY);
    if (fd >= 0)
      stream_ = _fdopen(fd, "rb");
    if (!stream_) {
      if (fd >= 0)
        _close(fd);
      else
        CloseHandle(read_pipe);
      TerminateProcess(process_, 1);
      WaitForSingleObject(process_, INFINITE);
      CloseHandle(process_);
      throw std::runtime_error("Cannot open child process pipe stream");
    }
#else
    int fd[2];
    if (pipe(fd) < 0)
      throw std::runtime_error("pipe failed");
    if (fcntl(fd[0], F_SETFD, FD_CLOEXEC) < 0 || fcntl(fd[1], F_SETFD, FD_CLOEXEC) < 0) {
      ::close(fd[0]);
      ::close(fd[1]);
      throw std::runtime_error("fcntl failed");
    }
    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_adddup2(&actions, fd[1], STDOUT_FILENO);
    posix_spawn_file_actions_addclose(&actions, fd[0]);
    posix_spawn_file_actions_addclose(&actions, fd[1]);
    std::vector<char *> argv;
    for (auto &arg : args)
      argv.push_back(arg.data());
    argv.push_back(nullptr);
    int status = posix_spawnp(&process_, argv[0], &actions, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    ::close(fd[1]);
    if (status) {
      ::close(fd[0]);
      process_ = -1;
      throw std::runtime_error(std::string("posix_spawnp: ") + std::strerror(status));
    }
    stream_ = fdopen(fd[0], "rb");
    if (!stream_) {
      ::close(fd[0]);
      kill(process_, SIGTERM);
      waitpid(process_, nullptr, 0);
      process_ = -1;
      throw std::runtime_error("fdopen failed");
    }
#endif
  }
  PipeProcess(const PipeProcess &) = delete;
  PipeProcess &operator=(const PipeProcess &) = delete;
  ~PipeProcess() {
    if (stream_)
      fclose(stream_);
#ifdef _WIN32
    if (process_) {
      if (WaitForSingleObject(process_, 0) == WAIT_TIMEOUT)
        TerminateProcess(process_, 1);
      WaitForSingleObject(process_, INFINITE);
      CloseHandle(process_);
    }
#else
    if (process_ > 0) {
      kill(process_, SIGTERM);
      while (waitpid(process_, nullptr, 0) < 0 && errno == EINTR) {
      }
    }
#endif
  }
  FILE *stream() const {
    return stream_;
  }
  void finish() {
#ifdef _WIN32
    DWORD code = 1;
    bool ok = WaitForSingleObject(process_, INFINITE) == WAIT_OBJECT_0 &&
              GetExitCodeProcess(process_, &code);
    CloseHandle(process_);
    process_ = nullptr;
    if (!ok || code != 0)
      throw std::runtime_error("Child process failed");
#else
    int status = 0;
    pid_t done;
    do {
      done = waitpid(process_, &status, 0);
    } while (done < 0 && errno == EINTR);
    process_ = -1;
    if (done < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
      throw std::runtime_error("Child process failed");
#endif
  }
};
} // namespace niji::platform
