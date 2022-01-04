collect_ignore = ["setup.py", "kn1d_example.py", "kn1d_wrapper.py", "docs/conf.py", "*julia*"]

# temporary, until issue with omfit_classes.omfit_eqdsk loading is clarified...
collect_ignore.append('test_basic.py')
collect_ignore.append('test_FSA.py')
