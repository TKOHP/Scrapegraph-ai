
import { atom, useAtomValue } from "jotai"

export function useLogin() {
  const login = () => {}
  const logout = () => {}

  return {
    loggedIn: false,
    userInfo: {},
    enableLogin: false,
    logout,
    login,
  }
}
