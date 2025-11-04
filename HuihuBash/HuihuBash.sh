#!/usr/bash

readonly CONNECTIVITY_URL="http://1.1.1.1"
readonly HUIHU_PWD="246135"
readonly HUIHU_ACC="17311177611"
readonly HUIHU_TELECOM="telecom"
readonly HUIHU_CHECK_INTERVAL_S=15

# -----------------------------------------------------------------------------
# huihu_auth
# -----------------------------------------------------------------------------
# Description:
#   Perform Huihu portal authentication by sending an HTTP POST to the portal
#   with the required JSON payload and headers.
#
# Globals used:
#   - HUIHU_TELECOM: Domain of the portal (e.g. "telecom", "cmcc").
#   - HUIHU_ACC:     Account/username used for authentication.
#   - HUIHU_PWD:     Password used for authentication.
#
# Returns (exit status):
#   0  AUTH_OK        - Request sent successfully (HTTP 200).
#   1  AUTH_ERR       - Authentication failed (non-200 response).
#   2  NOT_IN_NET     - Portal unreachable (HTTP code 000 or connection issue).
#   3  SOCK_ERR       - curl/network invocation error.
#
# Notes:
#   - This function prints a short status line to stdout/stderr for logging.
#   - The response body is not parsed; adapt the heuristics if the API provides
#     explicit success/error fields.
# -----------------------------------------------------------------------------
huihu_auth() {
    local url="http://10.10.16.12/api/portal/v1/login"

    # Build JSON payload. Avoid external dependencies for portability.
    local payload
    payload=$(printf '{"domain":"%s","username":"%s","password":"%s"}' \
        "$HUIHU_TELECOM" "$HUIHU_ACC" "$HUIHU_PWD")

    # Prepare curl options and request
    local resp
    if ! resp=$(curl --silent --show-error --location \
        -X POST \
        -H 'Accept: application/json, text/javascript, */*; q=0.01' \
        -H 'Accept-Encoding: gzip, deflate' \
        -H 'Accept-Language: zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6,es;q=0.5' \
        -H 'Connection: keep-alive' \
        -H 'Content-Type: application/json; charset=UTF-8' \
        -H 'Host: 10.10.16.12' \
        -H 'Origin: http://10.10.16.12' \
        -H 'Referer: http://10.10.16.12/portal/mobile.html?v=202208181518' \
        -H 'User-Agent: Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36 Edg/135.0.0.0' \
        -H 'X-Requested-With: XMLHttpRequest' \
        --data "$payload" \
        --write-out '\n%{http_code}' \
        "$url"); then
        echo "huihu_auth: curl invocation failed" >&2
        return 3
    fi

    # Separate body and HTTP code
    local http_code body
    http_code="${resp##*$'\n'}"
    body="${resp%$'\n'*}"

    echo "huihu_auth: HTTP $http_code"
    # Uncomment for verbose response logging
    # echo "huihu_auth: Response: $body"

    if [[ "$http_code" == "200" ]]; then
        echo "huihu_auth: Authentication request sent successfully"
        return 0
    fi

    # Heuristic: if curl returns HTTP code 000, the endpoint is unreachable
    if [[ "$http_code" == "000" ]]; then
        echo "huihu_auth: Portal unreachable (possibly not in campus network)" >&2
        return 2
    fi

    echo "huihu_auth: Authentication failed, HTTP $http_code" >&2
    return 1
}

while true; do
    curl --silent --head --fail "$CONNECTIVITY_URL" >/dev/null
    if [[ $? -eq 0 ]]; then
        # Network reachable; check again after interval
        sleep "$HUIHU_CHECK_INTERVAL_S"
    else
        echo "Connectivity check failed; attempting Huihu authentication..."
        if huihu_auth; then
            echo "Auth OK; will recheck connectivity after interval"
        else
            rc=$?
            case "$rc" in
            2)
                echo "Not in campus network; exiting."
                exit 0
                ;;
            3)
                echo "Socket/curl error during auth; will retry."
                ;;
            *)
                echo "Authentication error; will retry."
                ;;
            esac
        fi
        sleep "$HUIHU_CHECK_INTERVAL_S"
    fi
done
