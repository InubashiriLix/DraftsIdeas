---FUCKING JAVA LIKE SHIT OBJ
---
---@class Object
local Object = {}
Object.__index = Object

--- new function
---@param self Object
function Object:new() end

---@param self Object
---@return Object
function Object:extend()
    local obj = {}
    for k, v in pairs(self) do
        if k:find("^__") == 1 then
            obj[k] = v
        end
    end
    obj.__index = obj -- when properties not found in the instance, then lookup in the metatable first, then __index
    obj.super = self
    setmetatable(obj, self)
    return obj
end

---@param self Object
function Object:mixin(...)
    for _, cls in pairs({ ... }) do
        for k, v in pairs(cls) do
            if self[k] == nil and type(v) == "function" then
                self[k] = v
            end
        end
    end
end

---@param self Object
---@param T Object
---@return boolean
function Object:isinstanceof(T)
    local mt = getmetatable(self)

    while mt do
        if mt == T then
            return true
        end
        mt = getmetatable(mt)
    end

    return false
end

---@param self Object
---@return string
function Object:__tostring()
    return "Object"
end

function Object:__call(...)
    local obj = setmetatable({}, self)
    obj:new(...)
    return obj
end

return Object
