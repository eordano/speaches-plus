export interface NavOpts { replace?: boolean }
type NavFn = (to: string, opts?: NavOpts) => Promise<void>;

let navImpl: NavFn = () => Promise.resolve();
export const setNavigator = (fn: NavFn): void => { navImpl = fn; };
export const navTo = (to: string, opts?: NavOpts): Promise<void> => navImpl(to, opts);
