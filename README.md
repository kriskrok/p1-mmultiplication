# p1-mmultiplication exercise :cyclone:

Naive matrix multiplication script written in Python 3.

### Notes
The `requirements.txt` file lists all Python libraries needed for execution. These can be installed by chanting

```python
pip install -r requirements.txt
```

To run the multiplicator and collect CPU and memory usage information using psrecord run the `main.py` via 

```console
python3 main.py
```

and execute on a separate terminal 

```bash
psrecord pgrep -f '^python3 main.py$' --log psrecord_log.txt --plot psrecord_plot.png --interval 1
```

___
**_Happy multiplying!_**