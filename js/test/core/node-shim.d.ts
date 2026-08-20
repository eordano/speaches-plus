// Minimal ambient types so `tsc -p tsconfig.test.json` runs without @types/node
// (not installed in this workspace-only package). Delete once @types/node lands.
declare module "node:test" {
  export function test(name: string, fn: () => void | Promise<void>): Promise<void>;
}

declare module "node:assert/strict" {
  interface StrictAssert {
    (value: unknown, message?: string): asserts value;
    equal(actual: unknown, expected: unknown, message?: string): void;
    deepEqual(actual: unknown, expected: unknown, message?: string): void;
    ok(value: unknown, message?: string): asserts value;
    match(value: string, pattern: RegExp, message?: string): void;
    throws(fn: () => unknown, message?: string): void;
    rejects(block: Promise<unknown> | (() => Promise<unknown>), message?: string): Promise<void>;
    fail(message?: string): never;
  }
  const assert: StrictAssert;
  export default assert;
}
