#include "ggml-backend-dl.h"

#ifdef _WIN32

#include <cstdlib>
#include <mutex>
#include <string>

// GGML_BACKEND_DLL_DIRS: ';'-separated folders searched for a plugin's imports, e.g. a GPU vendor runtime installed apart from the plugin.
static void add_backend_dll_dirs() {
    static std::once_flag once;
    std::call_once(once, [] {
        const wchar_t * dirs = _wgetenv(L"GGML_BACKEND_DLL_DIRS");
        if (!dirs) {
            return;
        }
        std::wstring list(dirs);
        for (size_t start = 0; start <= list.size();) {
            size_t end = list.find(L';', start);
            if (end == std::wstring::npos) {
                end = list.size();
            }
            const std::wstring dir = list.substr(start, end - start);
            if (!dir.empty()) {
                AddDllDirectory(dir.c_str());
            }
            start = end + 1;
        }
    });
}

dl_handle * dl_load_library(const fs::path & path) {
    // suppress error dialogs for missing DLLs
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    add_backend_dll_dirs();

    // Resolve the plugin's own imports (e.g. a GPU vendor runtime shipped beside it) from the plugin's
    // folder first, then the application folder, GGML_BACKEND_DLL_DIRS and System32. PATH is not used (packaged apps ignore it).
    HMODULE handle = LoadLibraryExW(path.wstring().c_str(), nullptr,
                                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);

    SetErrorMode(old_mode);

    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
    SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

    void * p = (void *) GetProcAddress(handle, name);

    SetErrorMode(old_mode);

    return p;
}

const char * dl_error() {
    return "";
}

#else

dl_handle * dl_load_library(const fs::path & path) {
    dl_handle * handle = dlopen(path.string().c_str(), RTLD_NOW | RTLD_LOCAL);
    return handle;
}

void * dl_get_sym(dl_handle * handle, const char * name) {
    return dlsym(handle, name);
}

const char * dl_error() {
    const char *rslt = dlerror();
    return rslt != nullptr ? rslt : "";
}

#endif
