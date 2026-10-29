use anyhow::Result;

pub fn env_arm_opt_in_eq_some_1_reread_every_call(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

pub fn env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(name: &str) -> bool {
    std::env::var(name).ok().as_deref() != Some("0")
}

pub fn env_arm_present_at_all_reread_every_call(name: &str) -> bool {
    std::env::var_os(name).is_some()
}

pub fn env_arm_opt_in_set_to_anything_but_0_reread_every_call(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v != "0")
}

#[macro_export]
macro_rules! env_arm_opt_in_eq_some_1_cached_first_read_wins {
    ($name:expr) => {{
        static ON: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
        *ON.get_or_init(|| ::std::env::var($name).ok().as_deref() == Some("1"))
    }};
}

#[macro_export]
macro_rules! env_arm_opt_out_default_on_unless_eq_some_0_cached_first_read_wins {
    ($name:expr) => {{
        static ON: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
        *ON.get_or_init(|| ::std::env::var($name).ok().as_deref() != Some("0"))
    }};
}

#[macro_export]
macro_rules! env_arm_present_at_all_cached_first_read_wins {
    ($name:expr) => {{
        static ON: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
        *ON.get_or_init(|| ::std::env::var_os($name).is_some())
    }};
}

#[macro_export]
macro_rules! env_arm_opt_in_set_to_anything_but_0_cached_first_read_wins {
    ($name:expr) => {{
        static ON: ::std::sync::OnceLock<bool> = ::std::sync::OnceLock::new();
        *ON.get_or_init(|| ::std::env::var($name).is_ok_and(|v| v != "0"))
    }};
}

pub fn env_arm_dependency_assert_enabled_arm_requires_prerequisite(
    arm_env: &str,
    arm_on: bool,
    prerequisite_env: &str,
    prerequisite_on: bool,
    why_the_arm_is_wrong_without_it: &str,
) -> Result<()> {
    anyhow::ensure!(
        !arm_on || prerequisite_on,
        "{arm_env}=1 needs {prerequisite_env}=1: {why_the_arm_is_wrong_without_it}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_reread_env_arm_family_reads_fresh_every_call_with_its_own_truth_table() {
        let name = "NV_CFG_TEST_OPT_IN_REREAD";
        unsafe { std::env::remove_var(name) };
        assert!(!env_arm_opt_in_eq_some_1_reread_every_call(name));
        unsafe { std::env::set_var(name, "1") };
        assert!(env_arm_opt_in_eq_some_1_reread_every_call(name));
        unsafe { std::env::set_var(name, "0") };
        assert!(!env_arm_opt_in_eq_some_1_reread_every_call(name));
        unsafe { std::env::remove_var(name) };

        let name = "NV_CFG_TEST_OPT_OUT_REREAD";
        unsafe { std::env::remove_var(name) };
        assert!(env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(name));
        unsafe { std::env::set_var(name, "0") };
        assert!(!env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(name));
        unsafe { std::env::set_var(name, "off") };
        assert!(env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call(name));
        unsafe { std::env::remove_var(name) };

        let name = "NV_CFG_TEST_PRESENCE_REREAD";
        unsafe { std::env::remove_var(name) };
        assert!(!env_arm_present_at_all_reread_every_call(name));
        unsafe { std::env::set_var(name, "0") };
        assert!(env_arm_present_at_all_reread_every_call(name));
        unsafe { std::env::set_var(name, "") };
        assert!(env_arm_present_at_all_reread_every_call(name));
        unsafe { std::env::remove_var(name) };
        assert!(!env_arm_present_at_all_reread_every_call(name));

        let name = "NV_CFG_TEST_SET_NOT0_REREAD";
        unsafe { std::env::remove_var(name) };
        assert!(!env_arm_opt_in_set_to_anything_but_0_reread_every_call(
            name
        ));
        unsafe { std::env::set_var(name, "0") };
        assert!(!env_arm_opt_in_set_to_anything_but_0_reread_every_call(
            name
        ));
        unsafe { std::env::set_var(name, "off") };
        assert!(env_arm_opt_in_set_to_anything_but_0_reread_every_call(name));
        unsafe { std::env::set_var(name, "") };
        assert!(env_arm_opt_in_set_to_anything_but_0_reread_every_call(name));
        unsafe { std::env::remove_var(name) };
    }

    #[test]
    fn every_cached_env_arm_macro_caches_per_call_site_so_the_first_read_wins_over_later_env_changes(
    ) {
        let name = "NV_CFG_TEST_CACHED_FIRST_READ";
        fn one_call_site_like_a_family_wrapper_fn(name: &str) -> bool {
            env_arm_opt_in_eq_some_1_cached_first_read_wins!(name)
        }
        unsafe { std::env::set_var(name, "1") };
        assert!(one_call_site_like_a_family_wrapper_fn(name));
        unsafe { std::env::set_var(name, "0") };
        assert!(one_call_site_like_a_family_wrapper_fn(name));
        unsafe { std::env::remove_var(name) };
        assert!(!env_arm_opt_in_eq_some_1_cached_first_read_wins!(name));

        let name = "NV_CFG_TEST_CACHED_OPT_OUT";
        fn opt_out_call_site(name: &str) -> bool {
            env_arm_opt_out_default_on_unless_eq_some_0_cached_first_read_wins!(name)
        }
        unsafe { std::env::set_var(name, "0") };
        assert!(!opt_out_call_site(name));
        unsafe { std::env::remove_var(name) };
        assert!(!opt_out_call_site(name));
        assert!(env_arm_opt_out_default_on_unless_eq_some_0_cached_first_read_wins!(name));

        let name = "NV_CFG_TEST_CACHED_PRESENCE";
        fn presence_call_site(name: &str) -> bool {
            env_arm_present_at_all_cached_first_read_wins!(name)
        }
        unsafe { std::env::set_var(name, "0") };
        assert!(presence_call_site(name));
        unsafe { std::env::remove_var(name) };
        assert!(presence_call_site(name));
        assert!(!env_arm_present_at_all_cached_first_read_wins!(name));

        let name = "NV_CFG_TEST_CACHED_SET_NOT0";
        fn set_not0_call_site(name: &str) -> bool {
            env_arm_opt_in_set_to_anything_but_0_cached_first_read_wins!(name)
        }
        unsafe { std::env::set_var(name, "0") };
        assert!(!set_not0_call_site(name));
        unsafe { std::env::set_var(name, "on") };
        assert!(!set_not0_call_site(name));
        assert!(env_arm_opt_in_set_to_anything_but_0_cached_first_read_wins!(name));
        unsafe { std::env::remove_var(name) };
    }

    #[test]
    fn env_arm_dependency_assert_matches_the_shared_fold_needs_fused_chain_shape() {
        assert!(env_arm_dependency_assert_enabled_arm_requires_prerequisite(
            "NV_X_CHILD",
            false,
            "NV_X_PARENT",
            false,
            "unused"
        )
        .is_ok());
        assert!(env_arm_dependency_assert_enabled_arm_requires_prerequisite(
            "NV_X_CHILD",
            true,
            "NV_X_PARENT",
            true,
            "unused"
        )
        .is_ok());
        let err = env_arm_dependency_assert_enabled_arm_requires_prerequisite(
            "NV_X_CHILD",
            true,
            "NV_X_PARENT",
            false,
            "the unfused path would scatter with weight 1.0",
        )
        .unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("NV_X_CHILD=1 needs NV_X_PARENT=1"));
        assert!(msg.contains("scatter with weight 1.0"));
    }
}
