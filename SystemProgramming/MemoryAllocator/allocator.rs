/**
 * Custom Memory Allocator in Rust
 * High-performance allocator with pool allocation and bump allocation strategies
 */

use std::alloc::{GlobalAlloc, Layout};
use std::cell::UnsafeCell;
use std::ptr::{self, NonNull};
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

// ===== Bump Allocator =====
// Fast, simple allocator that bumps a pointer forward
pub struct BumpAllocator {
    heap_start: usize,
    heap_end: usize,
    next: AtomicUsize,
    allocations: AtomicUsize,
}

impl BumpAllocator {
    pub const fn new() -> Self {
        Self {
            heap_start: 0,
            heap_end: 0,
            next: AtomicUsize::new(0),
            allocations: AtomicUsize::new(0),
        }
    }

    pub unsafe fn init(&mut self, heap_start: usize, heap_size: usize) {
        self.heap_start = heap_start;
        self.heap_end = heap_start + heap_size;
        self.next.store(heap_start, Ordering::Relaxed);
    }

    pub fn reset(&mut self) {
        self.next.store(self.heap_start, Ordering::Relaxed);
        self.allocations.store(0, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for BumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let size = layout.size();
        let align = layout.align();

        // Align up
        let alloc_start = align_up(self.next.load(Ordering::Relaxed), align);
        let alloc_end = alloc_start.checked_add(size)?;

        if alloc_end > self.heap_end {
            return ptr::null_mut();
        }

        self.next.store(alloc_end, Ordering::Relaxed);
        self.allocations.fetch_add(1, Ordering::Relaxed);

        alloc_start as *mut u8
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        // Bump allocator doesn't free individual allocations
        self.allocations.fetch_sub(1, Ordering::Relaxed);
    }
}

// ===== Linked List Allocator =====
struct ListNode {
    size: usize,
    next: Option<&'static mut ListNode>,
}

impl ListNode {
    const fn new(size: usize) -> Self {
        ListNode { size, next: None }
    }

    fn start_addr(&self) -> usize {
        self as *const Self as usize
    }

    fn end_addr(&self) -> usize {
        self.start_addr() + self.size
    }
}

pub struct LinkedListAllocator {
    head: UnsafeCell<Option<&'static mut ListNode>>,
}

impl LinkedListAllocator {
    pub const fn new() -> Self {
        Self {
            head: UnsafeCell::new(None),
        }
    }

    pub unsafe fn init(&mut self, heap_start: usize, heap_size: usize) {
        self.add_free_region(heap_start, heap_size);
    }

    unsafe fn add_free_region(&mut self, addr: usize, size: usize) {
        assert_eq!(align_up(addr, std::mem::align_of::<ListNode>()), addr);
        assert!(size >= std::mem::size_of::<ListNode>());

        let mut node = ListNode::new(size);
        node.next = (*self.head.get()).take();
        let node_ptr = addr as *mut ListNode;
        node_ptr.write(node);
        *self.head.get() = Some(&mut *node_ptr);
    }

    fn find_region(&mut self, size: usize, align: usize) -> Option<(&'static mut ListNode, usize)> {
        let mut current = unsafe { (*self.head.get()).as_mut()? };

        loop {
            if let Ok(alloc_start) = Self::alloc_from_region(current, size, align) {
                let next = current.next.take();
                let ret = Some((current.clone(), alloc_start));
                unsafe { *self.head.get() = next };
                return ret;
            }

            if let Some(ref mut next) = current.next {
                current = next;
            } else {
                break;
            }
        }

        None
    }

    fn alloc_from_region(region: &ListNode, size: usize, align: usize) -> Result<usize, ()> {
        let alloc_start = align_up(region.start_addr(), align);
        let alloc_end = alloc_start.checked_add(size).ok_or(())?;

        if alloc_end > region.end_addr() {
            return Err(());
        }

        let excess_size = region.end_addr() - alloc_end;
        if excess_size > 0 && excess_size < std::mem::size_of::<ListNode>() {
            return Err(());
        }

        Ok(alloc_start)
    }

    fn size_align(layout: Layout) -> (usize, usize) {
        let layout = layout
            .align_to(std::mem::align_of::<ListNode>())
            .expect("adjusting alignment failed")
            .pad_to_align();
        let size = layout.size().max(std::mem::size_of::<ListNode>());
        (size, layout.align())
    }
}

unsafe impl GlobalAlloc for LinkedListAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let (size, align) = LinkedListAllocator::size_align(layout);

        if let Some((region, alloc_start)) = (*self.head.get()).as_mut()
            .and_then(|head| LinkedListAllocator::find_region(head, size, align))
        {
            let alloc_end = alloc_start.checked_add(size).expect("overflow");
            let excess_size = region.end_addr() - alloc_end;

            if excess_size > 0 {
                (*self.head.get()).as_mut().unwrap().add_free_region(alloc_end, excess_size);
            }

            alloc_start as *mut u8
        } else {
            ptr::null_mut()
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let (size, _) = LinkedListAllocator::size_align(layout);
        (*self.head.get()).as_mut().unwrap().add_free_region(ptr as usize, size);
    }
}

// ===== Pool Allocator =====
// Fixed-size block allocator for efficient small allocations
pub struct PoolAllocator<const BLOCK_SIZE: usize, const BLOCK_COUNT: usize> {
    free_list: AtomicPtr<PoolNode>,
    blocks: UnsafeCell<[[u8; BLOCK_SIZE]; BLOCK_COUNT]>,
}

struct PoolNode {
    next: *mut PoolNode,
}

impl<const BLOCK_SIZE: usize, const BLOCK_COUNT: usize> PoolAllocator<BLOCK_SIZE, BLOCK_COUNT> {
    pub const fn new() -> Self {
        Self {
            free_list: AtomicPtr::new(ptr::null_mut()),
            blocks: UnsafeCell::new([[0; BLOCK_SIZE]; BLOCK_COUNT]),
        }
    }

    pub unsafe fn init(&self) {
        let blocks = &mut *self.blocks.get();
        let mut prev: *mut PoolNode = ptr::null_mut();

        for block in blocks.iter_mut().rev() {
            let node = block.as_mut_ptr() as *mut PoolNode;
            (*node).next = prev;
            prev = node;
        }

        self.free_list.store(prev, Ordering::Relaxed);
    }

    unsafe fn pop(&self) -> Option<*mut u8> {
        loop {
            let head = self.free_list.load(Ordering::Acquire);
            if head.is_null() {
                return None;
            }

            let next = (*head).next;
            if self.free_list
                .compare_exchange(head, next, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return Some(head as *mut u8);
            }
        }
    }

    unsafe fn push(&self, ptr: *mut u8) {
        let node = ptr as *mut PoolNode;

        loop {
            let head = self.free_list.load(Ordering::Acquire);
            (*node).next = head;

            if self.free_list
                .compare_exchange(head, node, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }
}

unsafe impl<const BLOCK_SIZE: usize, const BLOCK_COUNT: usize> GlobalAlloc
    for PoolAllocator<BLOCK_SIZE, BLOCK_COUNT>
{
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() > BLOCK_SIZE || layout.align() > BLOCK_SIZE {
            return ptr::null_mut();
        }

        self.pop().unwrap_or(ptr::null_mut())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, _layout: Layout) {
        self.push(ptr);
    }
}

// ===== Utility Functions =====
fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

fn align_down(addr: usize, align: usize) -> usize {
    addr & !(align - 1)
}

// ===== Tests and Benchmarks =====
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bump_allocator() {
        let mut allocator = BumpAllocator::new();
        const HEAP_SIZE: usize = 1024 * 1024;
        let mut heap = vec![0u8; HEAP_SIZE];

        unsafe {
            allocator.init(heap.as_mut_ptr() as usize, HEAP_SIZE);

            let layout = Layout::from_size_align(128, 8).unwrap();
            let ptr1 = allocator.alloc(layout);
            assert!(!ptr1.is_null());

            let ptr2 = allocator.alloc(layout);
            assert!(!ptr2.is_null());
            assert_ne!(ptr1, ptr2);
        }
    }

    #[test]
    fn test_pool_allocator() {
        let allocator = PoolAllocator::<64, 100>::new();
        unsafe {
            allocator.init();

            let layout = Layout::from_size_align(32, 8).unwrap();
            let ptr = allocator.alloc(layout);
            assert!(!ptr.is_null());

            allocator.dealloc(ptr, layout);

            let ptr2 = allocator.alloc(layout);
            assert_eq!(ptr, ptr2);
        }
    }
}

fn main() {
    println!("Custom Memory Allocator Implementations");
    println!("1. Bump Allocator - Fast linear allocation");
    println!("2. Linked List Allocator - General purpose");
    println!("3. Pool Allocator - Fixed-size blocks");
}
