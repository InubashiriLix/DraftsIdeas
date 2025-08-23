#pragma once
#include <curl/curl.h>
#include <stdint.h>

#include <string>

#define DEBUG_

enum class NetConnStatus_E {
    FAILED = 0,
    OK = 1,
};

enum class AuthStatus_E {

    SOCK_ERR = -3,
    NOT_IN_NET = -2,
    AUTH_ERR = -1,
    OK = 0,
};

struct authInfo {
    std::string username;
    std::string password;
};

class HuihuFucker {
   public:
    HuihuFucker();
    void execute();
    NetConnStatus_E isNetConn();
    AuthStatus_E auth();

   private:
    bool readConfig();
    authInfo account;
    NetConnStatus_E netConnStatus;
    AuthStatus_E authStatus;
    uint32_t delay_time_checking = 30;
    uint32_t delay_time_auth_done = 60 * 60 * 24;
};
