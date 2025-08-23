#include <curl/curl.h>

#include <HuihuFucker.hpp>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <format>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <toml.hpp>
#include <workspaceConfig.hpp>
#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <unistd.h>
#endif

NetConnStatus_E HuihuFucker::isNetConn() {
    std::string command = "ping -c 1 180.76.76.76 > /dev/null 2>&1";
    int result = std::system(command.c_str());

    if (result == 0) {
        std::cout << "CheckConn: Network is reachable" << std::endl;
        return NetConnStatus_E::OK;
    } else {
        std::cout << "CheckConn: Network is unreachable" << std::endl;
        return NetConnStatus_E::FAILED;
    }
}

size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

AuthStatus_E HuihuFucker::auth() {
    CURL* curl;
    CURLcode res;
    std::string response_data;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, "http://10.10.16.12/api/portal/v1/login");

        // std::string post_fields =
        //     "{\"domain\":\"cmcc\",\"username\":\"18896759108\",\"password\":\"123321\"}";
        std::stringstream ss;
        ss << "{\"domain\":\"cmcc\",\"username\":\"" << this->account.username
           << "\",\"password\":\"" << this->account.password << "\"}";
        std::string post_fields = ss.str();

        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, post_fields.c_str());

        struct curl_slist* headers = nullptr;
        headers =
            curl_slist_append(headers, "Accept: application/json, text/javascript, */*; q=0.01");
        headers = curl_slist_append(headers, "Accept-Encoding: gzip, deflate");
        headers = curl_slist_append(
            headers, "Accept-Language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,es;q=0.5");
        headers = curl_slist_append(headers, "Connection: keep-alive");
        headers = curl_slist_append(headers, "Content-Type: application/json; charset=UTF-8");
        headers = curl_slist_append(headers, "Host: 10.10.16.12");
        headers = curl_slist_append(headers, "Origin: http://10.10.16.12");
        headers = curl_slist_append(
            headers, "Referer: http://10.10.16.12/portal/mobile.html?v=202208181518");
        headers = curl_slist_append(
            headers,
            "User-Agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 Edg/135.0.0.0");
        headers = curl_slist_append(headers, "X-Requested-With: XMLHttpRequest");

        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_data);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
            if (netConnStatus == NetConnStatus_E::OK) {
                return AuthStatus_E::NOT_IN_NET;
            }
            return AuthStatus_E::AUTH_ERR;
        } else {
            std::cout << "Auth: Authenticated successfully" << std::endl;
            std::cout << "Response Data: " << response_data << std::endl;
        }

        curl_slist_free_all(headers);

        curl_easy_cleanup(curl);
    } else {
        return AuthStatus_E::SOCK_ERR;
    }

    curl_global_cleanup();
    return AuthStatus_E::OK;
}
bool HuihuFucker::readConfig() {
#ifdef USE_PRODUCT_PATH
    const std::string config_toml_path = "../config.toml";
#else
    const std::string config_toml_path = "config.toml";
#endif
    // check the path existense
    auto temp = std::filesystem::path(config_toml_path);
    if (!std::filesystem::exists(temp)) {
        return false;
        // throw std::runtime_error("Config file not found");
    }
    auto config_data = toml::parse_file(config_toml_path);
    auto* db = config_data.as_table();

    auto username = (*db)["username"].value<std::string>();
    auto pwd = (*db)["pwd"].value<std::string>();

    if (username && pwd) {
        std::cout << "ReadData: get data successfully" << std::endl;
        std::cout << std::format("ReadData: username: {}", *username) << std::endl;
        std::cout << std::format("ReadData: password: {}", *pwd) << std::endl;
        this->account.username = *username;
        this->account.password = *pwd;
    } else {
        std::cerr << "ReadData: Error in reading data\n" << std::endl;
        return false;
    }

    // parse the data inside, and if non then raise error

    return true;
}

void HuihuFucker::execute() {
    uint32_t delay_time = 30;
    this->netConnStatus = this->isNetConn();
    this->authStatus = AuthStatus_E::AUTH_ERR;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(delay_time));
        // NOTE: reset the delay time all at this place (shit but effective)
        delay_time = this->delay_time_checking;
        this->netConnStatus = this->isNetConn();
        switch (this->netConnStatus) {
            case NetConnStatus_E::FAILED: {
                this->authStatus = this->auth();
                switch (this->authStatus) {
                    case AuthStatus_E::AUTH_ERR: {
                        // do nonthing
                        continue;
                    }
                    case AuthStatus_E::NOT_IN_NET: {
                        // exit
                        std::cout << "execute: Not in the school net, exiting";
                        exit(0);
                    }
                    case AuthStatus_E::OK: {
                        // reset the delay_time
                        std::cout << "execute: Auth DONE!" << std::endl;
                        delay_time = this->delay_time_auth_done;
                        // NOTE: if auth is done, then we can set a long time to check the net
                        continue;
                    }
                    case AuthStatus_E::SOCK_ERR: {
                        // ignore it
                        continue;
                    }
                }
            }

            case NetConnStatus_E::OK: {
                // if the net is connected, then just do nothing
                continue;
            }
            default: {
                std::cerr << "execute: switching net connection status: default case: INVALID"
                          << std::endl;
                exit(1);
            }
        }
    }
}

HuihuFucker::HuihuFucker() {
    if (!this->readConfig()) {
        std::cerr << "HuihuFucker: Error in reading config file\n" << std::endl;
        std::cerr << "Exiting" << std::endl;
        exit(1);
    }

    this->execute();
}
